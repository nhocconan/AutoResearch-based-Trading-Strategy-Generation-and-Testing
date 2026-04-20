#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Donchian_Breakout_VolumeTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Donchian Channels (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: 20-period high, lower band: 20-period low
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # === 12h Trend and Volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 30-period EMA for trend filter (15 days)
    close_series = pd.Series(close)
    ema30 = close_series.ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Volume ratio (10-period average)
    vol_series = pd.Series(volume)
    vol_ma10 = vol_series.rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / np.where(vol_ma10 > 0, vol_ma10, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        ema30_val = ema30[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(ema30_val) or 
            np.isnan(upper_val) or np.isnan(lower_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper band with volume and trend
            if (close_val > upper_val and 
                vol_ratio_val > 1.8 and
                close_val > ema30_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band with volume and trend
            elif (close_val < lower_val and 
                  vol_ratio_val > 1.8 and
                  close_val < ema30_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below EMA or volume drops
            if close_val < ema30_val or vol_ratio_val < 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above EMA or volume drops
            if close_val > ema30_val or vol_ratio_val < 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals