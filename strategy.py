#!/usr/bin/env python3
name = "12h_1d_VolumeWeighted_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's values (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_volume = np.roll(volume_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    prev_volume[0] = volume_1d[0]
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume ratio (volume spike detection)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 12h volume ratio (for entry confirmation)
    vol_ma_12h = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    vol_ratio_12h = volume / vol_ma_12h
    vol_ratio_12h = np.nan_to_num(vol_ratio_12h, nan=1.0)
    
    # Price channels: use previous day's range
    prev_range = prev_high - prev_low
    upper_channel = prev_high + 0.15 * prev_range  # 15% above prev high
    lower_channel = prev_low - 0.15 * prev_range   # 15% below prev low
    
    # Align price channels
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(vol_ratio_12h[i]) or np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filters: require both daily and 12h volume confirmation
        vol_filter = (vol_ratio_1d_aligned[i] > 1.4) and (vol_ratio_12h[i] > 1.3)
        
        if position == 0:
            # Long: Price breaks above upper channel with volume and bullish trend
            if (close[i] > upper_channel_aligned[i] and 
                vol_filter and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower channel with volume and bearish trend
            elif (close[i] < lower_channel_aligned[i] and 
                  vol_filter and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to channel or trend fails
            if position == 1:
                # Exit long: price returns to lower channel or trend turns bearish
                if (close[i] < lower_channel_aligned[i]) or (close[i] < ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to upper channel or trend turns bullish
                if (close[i] > upper_channel_aligned[i]) or (close[i] > ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals