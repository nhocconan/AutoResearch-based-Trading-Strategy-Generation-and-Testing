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
    
    # Get daily data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily volume average (20-period)
    volume_1d_series = pd.Series(volume_1d)
    vol_avg_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # 6h Donchian(20) breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    
    # Start after warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume confirmation: current 6h volume > 1.5x daily average volume (scaled)
        # Scale daily volume to 6h: 6h is 1/4 of daily volume
        vol_6h_expected = vol_avg_20_1d_aligned[i] / 4.0
        volume_confirm = volume[i] > 1.5 * vol_6h_expected
        
        if position == 0:
            # Long: price breaks above Donchian high + daily uptrend + volume
            if (price > high_20[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + daily downtrend + volume
            elif (price < low_20[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or daily trend turns down
            if (price < low_20[i] or 
                ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or daily trend turns up
            if (price > high_20[i] or 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_DailyEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0