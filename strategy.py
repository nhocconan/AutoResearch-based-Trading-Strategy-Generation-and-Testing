#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = np.full(len(high_1d), np.nan)
    lowest_low = np.full(len(low_1d), np.nan)
    
    for i in range(14, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-14:i+1])
        lowest_low[i] = np.min(low_1d[i-14:i+1])
    
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate 1d volume MA(10)
    vol_1d = df_1d['volume'].values
    vol_ma_10_1d = pd.Series(vol_1d).rolling(window=10, min_periods=10).mean().values
    
    # Align to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_10_1d)
    
    # Get 60-period EMA for trend filter (on 6h data)
    close_series = pd.Series(close)
    ema_60 = close_series.ewm(span=60, adjust=False, min_periods=60).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Williams %R and EMA
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_10_1d_aligned[i]) or 
            np.isnan(ema_60[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_10_1d_aligned[i]
        ema = ema_60[i]
        
        # Volume filter: volume > 1.3x 1d MA (volume confirmation)
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Entry conditions: Williams %R extremes with volume and trend
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above EMA + volume
            if wr < -80 and close[i] > ema and vol_filter:
                signals[i] = size
                position = 1
            # Short: Williams %R overbought (> -20) + price below EMA + volume
            elif wr > -20 and close[i] < ema and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or volume drops
            if wr > -50 or vol_now < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or volume drops
            if wr < -50 or vol_now < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsR_Volume_EMAFilter"
timeframe = "6h"
leverage = 1.0