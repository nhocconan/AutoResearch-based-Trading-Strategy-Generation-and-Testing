#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R3, 1d EMA34 up-trend, volume > 1.8x average
# Short when price breaks below S3, 1d EMA34 down-trend, volume > 1.8x average
# Exit when price reverts to Camarilla pivot point (mean reversion)
# Uses discrete position sizing (0.25) and tight volume filter to limit trades to ~15-35/year.
# Uses 1d for signal direction/trend, 6h only for entry timing and Camarilla levels.
# Targets 60-140 total trades over 4 years to avoid fee drag while capturing strong breakouts.
# Works in both bull and bear markets by following the higher timeframe trend.

name = "6h_Camarilla_R3S3_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 6h data for Camarilla levels (based on previous period)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Calculate 6h Camarilla levels using previous period
    # Camarilla: pivot = (H+L+C)/3, R3 = pivot + 1.1*(H-L), S3 = pivot - 1.1*(H-L)
    typical_price = (df_6h['high'] + df_6h['low'] + df_6h['close']) / 3.0
    high_low_range = df_6h['high'] - df_6h['low']
    camarilla_pivot = typical_price.shift(1).values  # previous period
    camarilla_range = high_low_range.shift(1).values  # previous period
    r3_level = camarilla_pivot + 1.1 * camarilla_range
    s3_level = camarilla_pivot - 1.1 * camarilla_range
    
    # Align 6h indicators to 6h timeframe (no additional delay needed for Camarilla)
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3_level)
    pivot_aligned = align_htf_to_ltf(prices, df_6h, camarilla_pivot)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Volume and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_pivot = pivot_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below pivot point (mean reversion)
            if curr_close < curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above pivot point (mean reversion)
            if curr_close > curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average (tight filter)
            vol_confirmed = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above R3, 1d EMA34 up-trend, volume confirmed
            if curr_high > curr_r3 and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, 1d EMA34 down-trend, volume confirmed
            elif curr_low < curr_s3 and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals