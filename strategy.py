#!/usr/bin/env python3
# 12h_1w_donchian_volume_breakout_v1
# Hypothesis: 12-hour Donchian(20) breakout with volume confirmation and 1-week EMA40 trend filter.
# Long: price > Donchian(20) high AND volume > 1.5x 20-period avg volume AND price > 1-week EMA40.
# Short: price < Donchian(20) low AND volume > 1.5x 20-period avg volume AND price < 1-week EMA40.
# Exit: price crosses Donchian midpoint or opposite breakout with volume.
# Designed for fewer trades (12-37/year) on 12h timeframe to reduce fee drag while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # 20-period average volume
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 1-week EMA40 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_40 = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 40:
        ema_1w_40[39] = np.mean(close_1w[:40])
        for i in range(40, len(close_1w)):
            ema_1w_40[i] = close_1w[i] * (2/41) + ema_1w_40[i-1] * (39/41)
    
    ema_1w_40_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_40)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        d_mid = donchian_mid[i]
        ema_1w = ema_1w_40_aligned[i]
        
        if np.isnan(d_high) or np.isnan(d_low) or np.isnan(d_mid) or np.isnan(avg_vol) or np.isnan(ema_1w):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.5 * avg_vol
        
        if position == 1:  # Long position
            if price < d_mid or (price < d_low and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            if price > d_mid or (price > d_high and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            if price > d_high and vol_surge and price > ema_1w:
                position = 1
                signals[i] = 0.30
            elif price < d_low and vol_surge and price < ema_1w:
                position = -1
                signals[i] = -0.30
    
    return signals