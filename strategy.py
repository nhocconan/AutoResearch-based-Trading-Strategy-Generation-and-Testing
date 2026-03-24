#!/usr/bin/env python3
"""
Experiment #900: 6h Primary + 1d/1w HTF — Fisher Transform + Dual HMA Trend + Volume Confirm

Hypothesis: 6h timeframe with daily AND weekly HTF bias provides superior trend filtering
vs single HTF. Fisher Transform excels at catching reversals in bear/range markets (2025 test).
Volume confirmation filters false breakouts. This combines proven edges:
1. Dual HTF trend (1d + 1w HMA) for robust direction filter
2. Fisher Transform for precise reversal entries (better than RSI in chop)
3. Volume spike confirmation to avoid fakeouts
4. ATR trailing stop for risk management

Key innovations:
1. 1w HMA(21) = major trend (only trade with weekly bias)
2. 1d HMA(21) = intermediate trend confirmation
3. 6h Fisher Transform(9) crosses ±1.5 for entry triggers
4. Volume > 1.5x SMA(20) confirms breakout validity
5. Discrete sizing: 0.0, ±0.25, ±0.30
6. 2.5x ATR trailing stop

Entry conditions (LOOSE to ensure trades):
- LONG: 1w HMA bull + 1d HMA bull + Fisher < -1.5 crossing up + Volume confirm
- SHORT: 1w HMA bear + 1d HMA bear + Fisher > +1.5 crossing down + Volume confirm

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_dual_hma_volume_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Transforms price into a Gaussian normal distribution
    Entry: Fisher crosses above -1.5 (long), below +1.5 (short)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            fisher[i] = 0.0
            fisher_signal[i] = 0.0
            continue
        
        x = 0.9999 * 2.0 * ((close[i] - lowest) / range_val - 0.5)
        x = np.clip(x, -0.999, 0.999)
        
        fisher_val = 0.5 * np.log((1 + x) / (1 - x))
        
        if i > period - 1 and not np.isnan(fisher[i - 1]):
            fisher_signal[i] = fisher[i - 1]
        else:
            fisher_signal[i] = fisher_val
        
        fisher[i] = fisher_val
    
    return fisher, fisher_signal

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
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    volume_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return volume_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher, fisher_signal = calculate_fisher(close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    volume_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
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
        
        if np.isnan(volume_sma_20[i]) or volume_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w + 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5
        fisher_long_cross = False
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
            fisher_long_cross = (fisher_signal[i-1] <= -1.5) and (fisher[i] > -1.5)
        
        # Short: Fisher crosses below +1.5
        fisher_short_cross = False
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
            fisher_short_cross = (fisher_signal[i-1] >= 1.5) and (fisher[i] < 1.5)
        
        # Also allow entries when Fisher is at extremes (looser condition)
        fisher_extreme_long = fisher[i] < -1.0
        fisher_extreme_short = fisher[i] > 1.0
        
        # === ENTRY LOGIC (LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_1w_bull and htf_1d_bull:
            if fisher_long_cross and volume_confirm:
                desired_signal = SIZE_STRONG
            elif fisher_extreme_long and volume_confirm:
                desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_1w_bear and htf_1d_bear:
            if fisher_short_cross and volume_confirm:
                desired_signal = -SIZE_STRONG
            elif fisher_extreme_short and volume_confirm:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals