#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_12hTrend_Volume_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (primary HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA34 on 12h close for trend
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h volume average for volume confirmation
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # Calculate Donchian channels on 4h data (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_ma_12h_aligned[i]) or \
           np.isnan(high_max[i]) or np.isnan(low_min[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current 4h volume > 1.5 * 12h average volume
        vol_confirm = volume[i] > (volume_ma_12h_aligned[i] * 1.5)
        
        if position == 0:
            # Long when price breaks above Donchian high with volume AND 12h trend is up
            if close[i] > high_max[i] and vol_confirm and close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low with volume AND 12h trend is down
            elif close[i] < low_min[i] and vol_confirm and close[i] < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below Donchian low or 12h trend turns down
            if close[i] < low_min[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above Donchian high or 12h trend turns up
            if close[i] > high_max[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals