#!/usr/bin/env python3
"""
Experiment #071: 4h Primary + 1d HTF — Fisher Transform Reversals + HMA Trend

Hypothesis: Fisher Transform excels at identifying reversal points in bear/range
markets (2022 crash, 2025 bear). Combined with 1d HMA for trend bias and ATR
volatility filter, this should generate 30-50 trades/year with better Sharpe
than KAMA+CRSI approach.

Key innovations:
1. Fisher Transform (Ehlers) - normalizes price to Gaussian, extremes = reversals
2. 1d HMA trend bias - simpler than KAMA, proven in best strategies
3. ATR ratio filter - only trade when vol > 1.2x recent avg (avoid dead markets)
4. Looser Fisher thresholds (-1.8/+1.8) to ensure sufficient trades
5. Dual exit: trailing stop + Fisher reversal signal

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Extremes (>1.5 or <-1.5) indicate high-probability reversals
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    # Calculate typical price and normalize
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Normalize price to 0-1 range
        value = (close[i] - lowest) / (highest - lowest)
        
        # Constrain to 0.001-0.999 to avoid log(0)
        value = max(0.001, min(0.999, value))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i - 1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother, less lag than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA calculations
    def wma(data, w):
        result = np.full(len(data), np.nan)
        for i in range(w - 1, len(data)):
            weights = np.arange(1, w + 1)
            result[i] = np.sum(data[i - w + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_input = 2 * wma_half - wma_full
    
    hma = wma(hma_input, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility filter and stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(atr, short_period=7, long_period=30):
    """ATR ratio - measures volatility expansion/contraction"""
    n = len(atr)
    if n < long_period:
        return np.full(n, np.nan)
    
    atr_short = pd.Series(atr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(atr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    ratio = np.full(n, np.nan)
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(atr, short_period=7, long_period=30)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_cross_bull = hma_fast[i] > hma_slow[i]
        hma_cross_bear = hma_fast[i] < hma_slow[i]
        
        # === VOLATILITY FILTER (ATR ratio > 1.1) ===
        vol_expanding = atr_ratio[i] > 1.1
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.8 from below (oversold reversal)
        fisher_long = fisher[i] > -1.8 and fisher_signal[i] <= -1.8
        
        # Short: Fisher crosses below +1.8 from above (overbought reversal)
        fisher_short = fisher[i] < 1.8 and fisher_signal[i] >= 1.8
        
        # === RSI CONFIRMATION (loose thresholds for more trades) ===
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TREND FOLLOWING: HTF + HMA cross + vol filter
        if htf_bull and hma_cross_bull and vol_expanding:
            desired_signal = SIZE_LONG
        elif htf_bear and hma_cross_bear and vol_expanding:
            desired_signal = -SIZE_SHORT
        
        # MEAN REVERSION: Fisher reversal with HTF bias (only when trend weak)
        hma_flat = abs(hma_fast[i] - hma_slow[i]) / close[i] < 0.008
        
        if hma_flat:
            if htf_bull and fisher_long and rsi_oversold:
                desired_signal = SIZE_LONG
            elif htf_bear and fisher_short and rsi_overbought:
                desired_signal = -SIZE_SHORT
        
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
        
        # Fisher reversal exit (opposite signal closes position)
        if in_position and position_side > 0 and fisher_short:
            stoploss_triggered = True
        if in_position and position_side < 0 and fisher_long:
            stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_LONG * 0.85:
            final_signal = SIZE_LONG
        elif desired_signal <= -SIZE_SHORT * 0.85:
            final_signal = -SIZE_SHORT
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