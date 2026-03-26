#!/usr/bin/env python3
"""
Experiment #011: 6h ATR Volatility Expansion + 1d Trend + Volume

HYPOTHESIS: ATR expansion beyond 1.2x its 30-bar average signals institutional 
involvement and momentum acceleration. Combined with 1d HMA trend alignment 
and volume confirmation, this catches major directional moves while avoiding 
ranging chop. Volatility expansion works bidirectionally:
- Uptrends: ATR spikes mark breakout acceleration points
- Downtrends: ATR spikes mark breakdown panic points (for shorts)

Why 6h: Slower than 4h, reducing fee drag. 1d HMA provides reliable multi-day 
trend direction without being too lagging. 3 conditions keep entries selective.

TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_atr_volatility_expansion_1d_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30_avg = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    
    # ATR expansion ratio (volatility breakout indicator)
    atr_expansion = atr_14 / np.where(atr_30_avg > 0, atr_30_avg, 1)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or np.isnan(atr_expansion[i]) or atr_expansion[i] <= 0:
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
        
        # === ENTRY CONDITIONS ===
        # 1. ATR expansion: current ATR > 1.2x 30-bar average
        atr_expanding = atr_expansion[i] > 1.2
        
        # 2. Volume confirmation: volume > 1.5x 20-bar average
        vol_spike = vol_ratio[i] > 1.5
        
        # 3. Trend direction from 1d HMA
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            if atr_expanding and vol_spike and price_above_1d_hma:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            elif atr_expanding and vol_spike and not price_above_1d_hma:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing stop) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT (3:1 R:R) ===
        if in_position and position_side > 0:
            profit_target = entry_price + 3.0 * entry_atr
            if high[i] >= profit_target:
                desired_signal = SIZE / 2  # Take partial profit, trail rest
        
        if in_position and position_side < 0:
            profit_target = entry_price - 3.0 * entry_atr
            if low[i] <= profit_target:
                desired_signal = -SIZE / 2  # Take partial profit, trail rest
        
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
        
        signals[i] = desired_signal
    
    return signals