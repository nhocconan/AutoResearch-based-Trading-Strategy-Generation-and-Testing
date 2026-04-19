#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Ehlers Fisher Transform with 1-day volume confirmation and ADX trend filter.
# Long when: Fisher crosses above -1.5, ADX(1d) > 20, volume > 1.5x 20-period average
# Short when: Fisher crosses below +1.5, ADX(1d) > 20, volume > 1.5x 20-period average
# Exit when Fisher crosses back through zero.
# Fisher Transform identifies turning points with Gaussian normal distribution, effective in both trending and ranging markets.
# Target: 15-25 trades/year per symbol. Uses proper smoothing to reduce noise.
name = "12h_FisherTransform_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on daily data using Wilder's smoothing
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) 
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # Directional Indicators
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Fisher Transform on 12h price (Ehlers)
    # Normalize price to [-1, 1] range over 10 periods
    def fisher_transform(price_series, length=10):
        n = len(price_series)
        if n < length:
            return np.full(n, np.nan)
        
        # Find highest high and lowest low over length period
        highest = np.full(n, np.nan)
        lowest = np.full(n, np.nan)
        
        for i in range(length-1, n):
            highest[i] = np.max(price_series[i-length+1:i+1])
            lowest[i] = np.min(price_series[i-length+1:i+1])
        
        # Avoid division by zero
        range_val = highest - lowest
        range_val = np.where(range_val == 0, 1e-10, range_val)
        
        # Normalize to [-1, 1]
        value = 2 * ((price_series - lowest) / range_val - 0.5)
        # Clip to prevent extreme values
        value = np.clip(value, -0.999, 0.999)
        
        # Fisher transform
        fisher = 0.5 * np.log((1 + value) / (1 - value))
        
        # Smooth with 3-period exponential smoothing
        smoothed = np.full(n, np.nan)
        if n >= 3:
            smoothed[2] = fisher[2]
            for i in range(3, n):
                smoothed[i] = 0.5 * fisher[i] + 0.5 * smoothed[i-1]
        
        return smoothed
    
    fisher = fisher_transform(close, length=10)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or np.isnan(fisher[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        fisher_val = fisher[i]
        fisher_prev = fisher[i-1] if i > 0 else 0
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Fisher crosses above -1.5, ADX > 20, volume confirmation
            if fisher_prev <= -1.5 and fisher_val > -1.5 and adx_val > 20 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Fisher crosses below +1.5, ADX > 20, volume confirmation
            elif fisher_prev >= 1.5 and fisher_val < 1.5 and adx_val > 20 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Fisher crosses below zero
            if fisher_prev > 0 and fisher_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Fisher crosses above zero
            if fisher_prev < 0 and fisher_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals