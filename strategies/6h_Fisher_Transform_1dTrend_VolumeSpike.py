#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Fisher_Transform_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Fisher Transform
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Fisher Transform on 1d close (period=9)
    # Fisher Transform formula: Fisher = 0.5 * ln((1 + X) / (1 - X))
    # where X = 2 * (price - min_low) / (max_high - min_low) - 1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate highest high and lowest low over 9 periods
    high_max = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_min = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    
    # Avoid division by zero
    range_hl = high_max - low_min
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    # Normalize price to [-1, 1]
    x_raw = 2 * (close_1d - low_min) / range_hl - 1
    # Clip to prevent log of negative or zero
    x_raw = np.clip(x_raw, -0.999, 0.999)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + x_raw) / (1 - x_raw))
    fisher_aligned = align_htf_to_ltf(prices, df_1d, fisher)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 34 for 1d EMA and 20 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(fisher_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        fisher_val = fisher_aligned[i]
        ema_1d = ema_34_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Fisher crosses above -1.5 AND price > 1d EMA34 (uptrend) AND volume > 2.0x average
            if fisher_val > -1.5 and close[i] > ema_1d and vol > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Fisher crosses below +1.5 AND price < 1d EMA34 (downtrend) AND volume > 2.0x average
            elif fisher_val < 1.5 and close[i] < ema_1d and vol > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Fisher crosses below -1.5 OR trend reverses (price < 1d EMA34)
            if fisher_val < -1.5 or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Fisher crosses above +1.5 OR trend reverses (price > 1d EMA34)
            if fisher_val > 1.5 or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals