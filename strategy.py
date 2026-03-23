#!/usr/bin/env python3
"""
Experiment #653: 1d Primary + 1w HTF — Simplified Dual Regime with Loose Entries

Hypothesis: Previous strategies failed due to OVER-FILTERING (0 trades on many exps).
This version SIMPLIFIES entry logic while keeping regime detection. Key changes:

1. LOOSER Fisher thresholds (-0.8/+0.8) to ensure trade frequency
2. Simplified regime: choppy=mean-revert, trending=trend-follow (no complex nested logic)
3. 1w HMA as SOFT bias only (doesn't block entries, just adjusts size)
4. Better hold logic: maintain position unless clear reversal signal
5. 2.5*ATR trailing stop for risk management
6. Discrete signal levels (0, ±0.25, ±0.30) to minimize fee churn

Why this should beat Sharpe=0.612:
- More trades = better statistical significance (fixes #1 failure mode)
- Regime detection prevents whipsaw in choppy markets
- 1d timeframe = lower fee drag, cleaner signals
- 1w HTF prevents major counter-trend disasters without over-filtering
- Conservative sizing (0.25-0.30) survives 77% crash with ~27% DD

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_fisher_kama_loose_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """Ehlers Fisher Transform for reversal detection."""
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    price = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        range_val = hh - ll
        if range_val < 1e-10:
            range_val = 1e-10
        
        price_raw = (close[i] - ll) / range_val
        
        if i > period:
            price[i] = 0.33 * 2 * (price_raw - 0.5) + 0.67 * price[i-1]
        else:
            price[i] = 0.33 * 2 * (price_raw - 0.5)
        
        price[i] = np.clip(price[i], -0.999, 0.999)
        fisher[i] = 0.5 * np.log((1 + price[i]) / (1 - price[i]))
        fisher_signal[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_signal

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    er = np.zeros(n)
    
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[er_period] = np.mean(close[:er_period+1])
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index for regime detection."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average for HTF trend."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators
    fisher_1d, fisher_signal_1d = calculate_fisher_transform(high, low, close, period=9)
    kama_1d = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_1d[i]) or np.isnan(fisher_signal_1d[i]):
            continue
        if np.isnan(kama_1d[i]) or np.isnan(chop_1d[i]):
            continue
        if np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop_1d[i] > 55.0
        is_trending = chop_1d[i] < 45.0
        
        # === HTF TREND BIAS (1w HMA) - SOFT FILTER ===
        htf_bullish = close[i] > hma_1w_aligned[i]
        htf_bearish = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_1d[i]
        kama_bearish = close[i] < kama_1d[i]
        
        # === FISHER SIGNALS (LOOSENED FOR FREQUENCY) ===
        fisher_long_cross = (fisher_1d[i] > -0.8) and (fisher_signal_1d[i] <= -0.8)
        fisher_short_cross = (fisher_1d[i] < 0.8) and (fisher_signal_1d[i] >= 0.8)
        
        fisher_oversold = fisher_1d[i] < -1.0
        fisher_overbought = fisher_1d[i] > 1.0
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY (Mean Reversion) ===
        if is_choppy:
            if fisher_oversold:
                desired_signal = SIZE_LONG
            elif fisher_overbought:
                desired_signal = -SIZE_SHORT
            elif fisher_long_cross:
                desired_signal = SIZE_LONG
            elif fisher_short_cross:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING (Trend Follow) ===
        elif is_trending:
            if kama_bullish and fisher_long_cross:
                desired_signal = SIZE_LONG if htf_bullish else SIZE_LONG * 0.8
            elif kama_bearish and fisher_short_cross:
                desired_signal = -SIZE_SHORT if htf_bearish else -SIZE_SHORT * 0.8
            # Also enter on KAMA cross with Fisher confirmation
            elif kama_bullish and fisher_1d[i] < 0.5:
                desired_signal = SIZE_LONG * 0.8
            elif kama_bearish and fisher_1d[i] > -0.5:
                desired_signal = -SIZE_SHORT * 0.8
        
        # === REGIME 3: NEUTRAL/TRANSITION ===
        else:
            if kama_bullish and fisher_1d[i] < 0.0:
                desired_signal = SIZE_LONG * 0.7
            elif kama_bearish and fisher_1d[i] > 0.0:
                desired_signal = -SIZE_SHORT * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC - Maintain position unless clear reversal ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish OR Fisher not extremely overbought
                if kama_bullish or fisher_1d[i] < 1.5:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if KAMA still bearish OR Fisher not extremely oversold
                if kama_bearish or fisher_1d[i] > -1.5:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.25:
            desired_signal = SIZE_LONG
        elif desired_signal > 0:
            desired_signal = SIZE_LONG * 0.8
        elif desired_signal < -0.20:
            desired_signal = -SIZE_SHORT
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT * 0.8
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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
        
        signals[i] = desired_signal
    
    return signals