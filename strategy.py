#!/usr/bin/env python3
"""
Experiment #025: 12h KAMA Trend + Volume Spike + 1d Trend Bias

HYPOTHESIS: KAMA adapts to volatility, better than fixed-period MAs for catching 
trend changes. Combined with volume spike confirmation (>1.5x) and 1d HMA 
trend bias, this captures institutional moves on 12h with minimal whipsaws.
Simple 3-condition entry: KAMA direction crosses + volume spike + HTF alignment.

TIMEFRAME: 12h
HTF: 1d for trend bias
TARGET: 50-100 total trades over 4 years (12-25/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_ema=2, slow_ema=30):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n - period)
    for i in range(n - period):
        vol = 0.0
        for j in range(period):
            vol += abs(close[period + i] - close[period + i - j])
        volatility[i] = vol if vol > 0 else 1e-10
    
    er = np.zeros(n)
    er[period:] = direction / volatility
    
    # Smooth constant
    fast_const = 2.0 / (fast_ema + 1)
    slow_const = 2.0 / (slow_ema + 1)
    const_sqrt = np.sqrt(fast_const / slow_const)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        sc = (er[i] * (fast_const - slow_const) + slow_const) * const_sqrt
        kama[i] = kama[i - 1] + sc * sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # KAMA(10) for trend detection
    kama_10 = calculate_kama(close, period=10)
    
    # Previous KAMA for crossover detection
    kama_10_prev = np.roll(kama_10, 1)
    kama_10_prev[0] = np.nan
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(kama_10_prev[i]):
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
        
        # === KAMA CROSSOVER DETECTION ===
        kama_cross_up = (close[i] > kama_10[i]) and (close[i-1] <= kama_10_prev[i-1])
        kama_cross_down = (close[i] < kama_10[i]) and (close[i-1] >= kama_10_prev[i-1])
        
        # === 1d TREND BIAS ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # KAMA cross up + volume spike + bullish 1d trend
            if kama_cross_up and vol_spike and price_above_1d_hma:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # KAMA cross down + volume spike + bearish 1d trend
            if kama_cross_down and vol_spike and not price_above_1d_hma:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (3 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: KAMA cross down + bearish 1d
            if kama_cross_down and not price_above_1d_hma:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: KAMA cross up + bullish 1d
            if kama_cross_up and price_above_1d_hma:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
            else:
                # Same direction - maintain position (no churn)
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals