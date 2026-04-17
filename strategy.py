#!/usr/bin/env python3
"""
1d Donchian Breakout + 1w EMA34 Trend + Volume Spike
Long: Close > Donchian(20) high + Close > 1w EMA34 + Volume > 1.5x 1d volume SMA(20)
Short: Close < Donchian(20) low + Close < 1w EMA34 + Volume > 1.5x 1d volume SMA(20)
Exit: Opposite Donchian break or trend filter fails
Targets 1d timeframe with 1w trend filter for multi-timeframe alignment.
Designed to capture strong trending moves with volume confirmation.
Target: 30-100 total trades over 4 years (7-25/year)
"""

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
    
    # Get 1d data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume SMA(20)
    volume_sma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian channels
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_sma[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        vol_sma_val = volume_sma[i]
        ema_trend = ema_34_1w_aligned[i]
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        
        if position == 0:
            # Long: Break above Donchian high + above 1w EMA34 + volume spike
            if price > d_high and price > ema_trend and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + below 1w EMA34 + volume spike
            elif price < d_low and price < ema_trend and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Break below Donchian low OR price below 1w EMA34
            if price < d_low or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Break above Donchian high OR price above 1w EMA34
            if price > d_high or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0