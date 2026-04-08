#!/usr/bin/env python3
# 4h_1d_donchian_volume_breakout_v4
# Hypothesis: 4-hour Donchian breakout with volume confirmation and 1-day EMA trend filter.
# Long: price > Donchian(20) high AND volume > 1.5x 20-period average volume AND price > 1-day EMA50.
# Short: price < Donchian(20) low AND volume > 1.5x 20-period average volume AND price < 1-day EMA50.
# Exit: price crosses Donchian midpoint or opposite breakout with volume.
# Designed to capture strong trending moves in both bull and bear markets with strict entry criteria to limit trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_breakout_v4"
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
    
    # 4h Donchian channels (20-period)
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
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_50 = np.full(len(close_1d), np.nan)
    ema_1d_50[49] = np.mean(close_1d[:50])
    for i in range(50, len(close_1d)):
        ema_1d_50[i] = close_1d[i] * (2/51) + ema_1d_50[i-1] * (49/51)
    
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        d_mid = donchian_mid[i]
        ema_1d = ema_1d_50_aligned[i]
        
        if np.isnan(d_high) or np.isnan(d_low) or np.isnan(d_mid) or np.isnan(avg_vol) or np.isnan(ema_1d):
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
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price > d_mid or (price > d_high and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price > d_high and vol_surge and price > ema_1d:
                position = 1
                signals[i] = 0.25
            elif price < d_low and vol_surge and price < ema_1d:
                position = -1
                signals[i] = -0.25
    
    return signals