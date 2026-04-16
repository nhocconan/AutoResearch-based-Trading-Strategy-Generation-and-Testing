#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout (20) for direction and 1d EMA50 for trend filter.
# Enter long when price breaks above 4h Donchian upper AND close > 1d EMA50 AND volume > 1.5x 20-period average.
# Enter short when price breaks below 4h Donchian lower AND close < 1d EMA50 AND volume > 1.5x 20-period average.
# Exit when price returns to 4h Donchian midpoint.
# Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Discrete position size 0.20 to minimize fee churn. Target: 60-150 total trades over 4 years (15-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead and TypeError
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Get 1d data once before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian Channel (20-period) on 4h
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Align Donchian levels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    midpoint_aligned = align_htf_to_ltf(prices, df_4h, midpoint_20)
    
    # Volume moving average (20-period) on 4h
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        ema_50_val = ema_50_aligned[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        midpoint_val = midpoint_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to midpoint
            if price <= midpoint_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to midpoint
            if price >= midpoint_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma_val
            
            # Trend filter: price relative to 1d EMA50
            trend_filter_long = price > ema_50_val
            trend_filter_short = price < ema_50_val
            
            # LONG: price breaks above Donchian upper with volume and trend confirmation
            if price > upper_val and vol_filter and trend_filter_long:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower with volume and trend confirmation
            elif price < lower_val and vol_filter and trend_filter_short:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_Donchian20_1dEMA50_Volume_SessionFilter_V1"
timeframe = "1h"
leverage = 1.0