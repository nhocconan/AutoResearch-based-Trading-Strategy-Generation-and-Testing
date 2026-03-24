#!/usr/bin/env python3
"""
Experiment #018: 30m Primary + 4h/1d HTF — Regime-Adaptive Mean Reversion with Session Filter

Hypothesis: 30m strategies fail due to either (1) too many trades → fee drag, or (2) too strict filters → 0 trades.
This uses PROVEN patterns with LOOSE thresholds to ensure trade generation:
1. 4h HMA for trend BIAS (not hard filter) - determines long/short preference
2. 1d HMA for regime confirmation - only trade WITH 1d trend in choppy markets
3. Connors RSI with LOOSE thresholds (15/85) - ensures ≥30 trades/symbol
4. Volume filter (>0.8x 20-period avg) - confirms participation
5. Session filter (8-20 UTC) - avoids low-liquidity hours
6. ATR trailing stop at 2.5x for risk management

Key improvements from failed 30m attempts:
- LOOSE CRSI thresholds (15/85 vs 10/90) to ensure trade generation
- 4h/1d HMA for BIAS only, not hard entry filter
- Volume + session filters reduce false signals without killing trade count
- Discrete signal levels (0.0, ±0.20, ±0.30) minimize fee churn
- Size: 0.20-0.30 (smaller for lower TF to control DD)

Entry Logic:
- CHOPPY (CHOP > 55): CRSI < 15 long, CRSI > 85 short (mean reversion)
- TRENDING (CHOP < 45): Pullback to 4h HMA + CRSI confirmation
- Volume > 0.8x average, Session 8-20 UTC
- Size: 0.30 with HTF trend, 0.20 against HTF trend

Risk: 2.5x ATR trailing stop, max signal magnitude 0.35
Target: Sharpe > 0.4, trades > 30/symbol train, > 3/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_4h1d_session_volume_v2"
timeframe = "30m"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - 3-component mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Entry: CRSI < 15-20 = oversold (long), CRSI > 80-85 = overbought (short)
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
    """Hull Moving Average - faster response than EMA"""
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
    """
    Choppiness Index - regime detection
    CHOP > 55 = choppy/range (mean revert)
    CHOP < 45 = trending (trend follow)
    """
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

def calculate_volume_avg(volume, period=20):
    """Simple volume average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_avg[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_avg

def get_hour_from_timestamp(open_time):
    """Extract UTC hour from timestamp (milliseconds)"""
    # Timestamp is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close)) * 1800000
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_timestamp(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === HTF TREND BIAS ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # Only trade during session hours for liquidity
        if not in_session:
            signals[i] = 0.0
            if in_position:
                # Hold position but don't add
                if position_side > 0:
                    highest_since_entry = max(highest_since_entry, close[i])
                elif position_side < 0:
                    lowest_since_entry = min(lowest_since_entry, close[i])
            continue
        
        if is_choppy:
            # MEAN REVERSION REGIME
            # Long: CRSI < 15 (oversold) + volume ok
            if crsi[i] < 15.0 and volume_ok:
                if hma_1d_bull:  # With 1d trend
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = signal_strength
            
            # Short: CRSI > 85 (overbought) + volume ok
            elif crsi[i] > 85.0 and volume_ok:
                if hma_1d_bear:  # With 1d trend
                    signal_strength = BASE_SIZE
                else:
                    signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength
        
        elif is_trending:
            # TREND REGIME - pullback entries
            # Long: 4h bullish + pullback (CRSI < 40)
            if hma_4h_bull and hma_1d_bull and crsi[i] < 40.0 and volume_ok:
                signal_strength = BASE_SIZE
                desired_signal = signal_strength
            
            # Short: 4h bearish + pullback (CRSI > 60)
            elif hma_4h_bear and hma_1d_bear and crsi[i] > 60.0 and volume_ok:
                signal_strength = BASE_SIZE
                desired_signal = -signal_strength
        
        else:
            # NEUTRAL REGIME (45 <= CHOP <= 55) - only trade WITH 1d trend
            if hma_1d_bull and crsi[i] < 25.0 and volume_ok:
                desired_signal = REDUCED_SIZE
            elif hma_1d_bear and crsi[i] > 75.0 and volume_ok:
                desired_signal = -REDUCED_SIZE
        
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