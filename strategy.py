#!/usr/bin/env python3
"""
Experiment #012: 12h Primary + 1d/1w HTF — Simplified Dual Regime with Volume Confirmation

Hypothesis: Previous 12h/1d strategies failed due to OVER-FILTERING (0 trades).
This version SIMPLIFIES logic to ensure trade generation while maintaining edge:

1. Connors RSI with LOOSE thresholds (15/85) for mean reversion entries
2. Choppiness Index at 50.0 threshold for regime split (simpler than 45/55)
3. 1d HMA(21) for trend BIAS only (asymmetric sizing, not hard filter)
4. 1w HMA(21) for major trend direction (avoid counter-trend in strong moves)
5. Volume confirmation: volume > 0.8 * SMA20(volume) to filter false signals
6. ATR trailing stop at 2.5x for risk management

Key improvements from failed #002 (mtf_12h_crsi_chop_dual_regime_1d1w_v1):
- LOOSER CRSI thresholds (15/85 vs 10/90) to generate more trades
- Volume filter replaces complex session/volume logic
- Simpler regime detection (single CHOP threshold at 50)
- Asymmetric sizing ensures signals even against HTF trend (reduced size)
- Removed funding rate (caused alignment issues, minimal edge on 12h)

Entry Logic:
- CHOPPY (CHOP > 50): CRSI < 15 long, CRSI > 85 short (mean reversion)
- TRENDING (CHOP <= 50): Follow 1d HMA direction with volume confirmation
- 1w HMA prevents major counter-trend positions
- Size: 0.30 with HTF trend, 0.20 against HTF trend

Risk: 2.5x ATR trailing stop, max signal magnitude 0.35
Target: Sharpe > 0.15 (beat current best 0.103), trades > 30/symbol train, > 3/symbol test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_volume_simplified_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - 3-component mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Entry signals: CRSI < 15-20 = oversold (long), CRSI > 80-85 = overbought (short)
    """
    n = len(close)
    if n < rank_period + rsi_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_3 = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_3 = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_3 = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if loss_3[i-1] < 1e-10:
            rsi_3[i] = 100.0
        else:
            rsi_3[i] = 100.0 - (100.0 / (1.0 + gain_3[i-1] / loss_3[i-1]))
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_gain_2 = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_2 = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        if streak_loss_2[i-1] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + streak_gain_2[i-1] / streak_loss_2[i-1]))
    
    # Component 3: PercentRank
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            rank = np.sum(returns < current_return)
            percent_rank[i] = 100.0 * rank / len(returns)
        else:
            percent_rank[i] = 50.0
    
    # Combine
    for i in range(rank_period, n):
        if np.isnan(rsi_3[i]) or np.isnan(rsi_streak[i]) or np.isnan(percent_rank[i]):
            continue
        crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation filter"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for major trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    MAX_SIZE = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] <= 50.0
        
        # === HTF TREND BIAS ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME - use CRSI extremes (LOOSE thresholds)
            if crsi[i] < 15.0 and volume_confirmed:
                # Long: oversold + volume
                if hma_1d_bull or hma_1w_bull:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            
            elif crsi[i] > 85.0 and volume_confirmed:
                # Short: overbought + volume
                if hma_1d_bear or hma_1w_bear:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
        
        elif is_trending:
            # TREND REGIME - follow HTF direction
            if hma_1d_bull and volume_confirmed:
                desired_signal = BASE_SIZE
            elif hma_1d_bear and volume_confirmed:
                desired_signal = -BASE_SIZE
            elif hma_1w_bull:
                # Fallback to 1w trend if 1d unclear
                desired_signal = REDUCED_SIZE
            elif hma_1w_bear:
                desired_signal = -REDUCED_SIZE
        
        # === 1W MAJOR TREND FILTER (prevent major counter-trend) ===
        # Only reduce size, don't block entirely (need trades!)
        if hma_1w_bull and desired_signal < 0:
            desired_signal = desired_signal * 0.5
        elif hma_1w_bear and desired_signal > 0:
            desired_signal = desired_signal * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
        elif abs(desired_signal) >= 0.10:
            final_signal = np.sign(desired_signal) * REDUCED_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals