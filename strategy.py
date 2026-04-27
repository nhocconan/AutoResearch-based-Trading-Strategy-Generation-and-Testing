#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d timeframe for directional bias with 1h volatility breakout entries.
# Uses 4h ADX > 25 for trend strength, 1d Supertrend for direction, and 1h ATR breakout for entry timing.
# Includes session filter (08-20 UTC) to reduce noise. Target: 15-37 trades/year (60-150 over 4 years).
# Position size: 0.20 (20% of capital) to manage drawdown in volatile markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX (14) on 4h
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        def ma(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) >= period:
                result[period-1] = np.nanmean(arr[:period])
                for i in range(period, len(arr)):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_ma = ma(tr, period)
        dm_plus_ma = ma(dm_plus, period)
        dm_minus_ma = ma(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_ma / tr_ma
        di_minus = 100 * dm_minus_ma / tr_ma
        
        # DX and ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        adx = ma(dx, period)
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Get 1d data for Supertrend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Supertrend (10, 3.0) on 1d
    def calculate_supertrend(high, low, close, atr_period=10, multiplier=3.0):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR
        atr = np.full_like(tr, np.nan)
        for i in range(len(tr)):
            if i < atr_period:
                atr[i] = np.nan
            elif i == atr_period:
                atr[i] = np.nanmean(tr[1:i+1])
            else:
                atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
        
        # Basic Bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Final Bands
        final_upper = np.full_like(upper_band, np.nan)
        final_lower = np.full_like(lower_band, np.nan)
        
        for i in range(len(close)):
            if np.isnan(atr[i]):
                final_upper[i] = np.nan
                final_lower[i] = np.nan
                continue
                
            if i == 0:
                final_upper[i] = upper_band[i]
                final_lower[i] = lower_band[i]
            else:
                if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                    final_upper[i] = upper_band[i]
                else:
                    final_upper[i] = final_upper[i-1]
                    
                if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                    final_lower[i] = lower_band[i]
                else:
                    final_lower[i] = final_lower[i-1]
        
        # Supertrend
        supertrend = np.full_like(close, np.nan)
        for i in range(len(close)):
            if np.isnan(atr[i]) or np.isnan(final_upper[i]) or np.isnan(final_lower[i]):
                supertrend[i] = np.nan
            elif i == 0:
                supertrend[i] = final_upper[i]
            else:
                if supertrend[i-1] == final_upper[i-1] and close[i] <= final_upper[i]:
                    supertrend[i] = final_upper[i]
                elif supertrend[i-1] == final_upper[i-1] and close[i] > final_upper[i]:
                    supertrend[i] = final_lower[i]
                elif supertrend[i-1] == final_lower[i-1] and close[i] >= final_lower[i]:
                    supertrend[i] = final_lower[i]
                elif supertrend[i-1] == final_lower[i-1] and close[i] < final_lower[i]:
                    supertrend[i] = final_upper[i]
        return supertrend
    
    supertrend_1d = calculate_supertrend(high_1d, low_1d, close_1d, 10, 3.0)
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    
    # Get 1h ATR for entry timing
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        for i in range(len(tr)):
            if i < period:
                atr[i] = np.nan
            elif i == period:
                atr[i] = np.nanmean(tr[1:i+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1h = calculate_atr(high, low, close, 14)
    
    # Calculate 1h moving average for breakout
    def calculate_ma(close, period):
        ma = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i < period - 1:
                ma[i] = np.nan
            else:
                ma[i] = np.mean(close[i-period+1:i+1])
        return ma
    
    ma_1h = calculate_ma(close, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup period
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(supertrend_1d_aligned[i]) or 
            np.isnan(atr_1h[i]) or np.isnan(ma_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        
        # Trend filter: 4h ADX > 25 indicates strong trend
        trend_filter = adx_4h_aligned[i] > 25
        
        # Direction: 1d Supertrend
        direction_long = price > supertrend_1d_aligned[i]
        direction_short = price < supertrend_1d_aligned[i]
        
        # Entry: 1h price breaks above/below MA with ATR buffer
        long_entry = price > ma_1h[i] + 0.5 * atr
        short_entry = price < ma_1h[i] - 0.5 * atr
        
        # Exit: reverse signal or volatility expansion
        long_exit = price < ma_1h[i] - 0.5 * atr
        short_exit = price > ma_1h[i] + 0.5 * atr
        
        if position == 0 and trend_filter:
            # Long entry
            if direction_long and long_entry:
                signals[i] = size
                position = 1
            # Short entry
            elif direction_short and short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit
            if long_exit or not direction_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit
            if short_exit or not direction_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_ADX_Supertrend_ATR_Breakout"
timeframe = "1h"
leverage = 1.0