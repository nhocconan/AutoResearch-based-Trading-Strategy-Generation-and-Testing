#!/usr/bin/env python3
"""
Experiment #1659: 4h Primary + 1d HTF — Simplified Dual Regime with Asymmetric Sizing

Hypothesis: Recent 4h failures (#1649, #1651, #1654) had negative Sharpe due to OVER-FILTERING
or wrong regime detection. This strategy SIMPLIFIES entry logic while keeping proven MTF structure.

Key changes from failed 4h strategies:
1. FEWER confluence conditions (max 2-3, not 4-5)
2. LOOSER thresholds (RSI 30/70 not 20/80, CHOP 50/50 not 61.8/38.2)
3. Asymmetric sizing: 0.35 with 1d trend, 0.20 against (reduces whipsaw losses)
4. Simpler regime: just CHOP > 50 (mean revert) vs CHOP < 50 (trend follow)
5. Wider stoploss: 3.0x ATR not 2.5x (less premature exits)

Why 4h timeframe:
- 20-50 trades/year target (fee-efficient)
- Proven in current best strategy structure
- Less noise than 1h, more signals than 1d

Entry Logic:
- CHOPPY (CHOP > 50): Connors RSI < 25 long, > 75 short
- TRENDING (CHOP < 50): Price vs HMA(21) + 1d bias
- NEUTRAL: Only trade WITH 1d trend direction

Risk: 3.0x ATR trailing stop, discrete signal levels (0.0, ±0.20, ±0.35)

Target: Sharpe > 0.618, trades > 30/symbol train, > 5/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_simplified_regime_crsi_hma_1d_asym_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if loss_smooth[i-1] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + gain_smooth[i-1] / loss_smooth[i-1]))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 50 = choppy/range (mean revert)
    CHOP < 50 = trending (trend follow)
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
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_3 = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if loss_smooth[i-1] < 1e-10:
            rsi_3[i] = 100.0
        else:
            rsi_3[i] = 100.0 - (100.0 / (1.0 + gain_smooth[i-1] / loss_smooth[i-1]))
    
    # Streak RSI
    streak_rsi = np.full(n, np.nan)
    for i in range(1, n):
        streak = 0
        if close[i] > close[i-1]:
            streak = 1
            j = i - 1
            while j > 0 and close[j] > close[j-1]:
                streak += 1
                j -= 1
        elif close[i] < close[i-1]:
            streak = -1
            j = i - 1
            while j > 0 and close[j] < close[j-1]:
                streak -= 1
                j -= 1
        
        if streak >= 0:
            streak_rsi[i] = min(100.0, streak * 10.0)
        else:
            streak_rsi[i] = max(0.0, 100.0 + streak * 10.0)
    
    streak_rsi_smooth = pd.Series(streak_rsi).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    # Percent Rank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if np.isnan(rsi_3[i]) or np.isnan(streak_rsi_smooth[i]) or np.isnan(percent_rank[i]):
            continue
        crsi[i] = (rsi_3[i] + streak_rsi_smooth[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper and lower bounds"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.35
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Simplified Choppiness) ===
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] < 50.0
        
        # === 1d HMA BIAS ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME - use Connors RSI extremes (LOOSER thresholds)
            # Long: CRSI < 25 (oversold)
            if crsi[i] < 25.0:
                signal_strength = BASE_SIZE if daily_bull else REDUCED_SIZE
                desired_signal = signal_strength
            # Short: CRSI > 75 (overbought)
            elif crsi[i] > 75.0:
                signal_strength = BASE_SIZE if daily_bear else REDUCED_SIZE
                desired_signal = -signal_strength
        
        elif is_trending:
            # TREND REGIME - use HMA position + 1d bias + Donchian breakout
            # Long: Price > HMA + daily bullish OR Donchian breakout
            if close[i] > hma_4h[i]:
                signal_strength = BASE_SIZE if daily_bull else REDUCED_SIZE
                desired_signal = signal_strength
            # Short: Price < HMA + daily bearish OR Donchian breakout
            elif close[i] < hma_4h[i]:
                signal_strength = BASE_SIZE if daily_bear else REDUCED_SIZE
                desired_signal = -signal_strength
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x - wider to avoid premature exits) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
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