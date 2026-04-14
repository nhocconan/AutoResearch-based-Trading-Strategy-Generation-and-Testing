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
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate daily ATR(14) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        
        # Get previous day's data (1d index)
        if i >= 1:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            
            # Calculate pivot points (standard formula)
            pivot = (prev_high + prev_low + prev_close) / 3.0
            s1 = (2 * pivot) - prev_high
            r1 = (2 * pivot) - prev_low
            s2 = pivot - (prev_high - prev_low)
            r2 = pivot + (prev_high - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            r3 = prev_high + 2 * (pivot - prev_low)
            
            # Align S3/R3 to daily timeframe (constant values for the day)
            s3_array = np.full(len(df_1d), s3)
            r3_array = np.full(len(df_1d), r3)
            s3_1d = align_htf_to_ltf(prices, df_1d, s3_array)[i]
            r3_1d = align_htf_to_ltf(prices, df_1d, r3_array)[i]
            
            if position == 0:
                # Long: Price breaks above R3 with volume and trend confirmation
                vol_ma = np.mean(volume[i-5:i]) if i >= 5 else volume[i]
                if (close[i] > r3_1d and close[i-1] <= r3_1d and 
                    volume[i] > vol_ma * 1.5 and 
                    close[i] > ema50_1d_aligned[i]):
                    position = 1
                    signals[i] = position_size
                # Short: Price breaks below S3 with volume and trend confirmation
                elif (close[i] < s3_1d and close[i-1] >= s3_1d and 
                      volume[i] > vol_ma * 1.5 and 
                      close[i] < ema50_1d_aligned[i]):
                    position = -1
                    signals[i] = -position_size
            elif position == 1:
                # Exit: Price breaks below S3 or trend reversal (price below EMA50)
                if close[i] < s3_1d or close[i] < ema50_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
            elif position == -1:
                # Exit: Price breaks above R3 or trend reversal (price above EMA50)
                if close[i] > r3_1d or close[i] > ema50_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "12h_Pivot_S3R3_Breakout_EMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0