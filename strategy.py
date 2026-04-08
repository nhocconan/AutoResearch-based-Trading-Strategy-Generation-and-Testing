#!/usr/bin/env python3
# 4h_1d_atr_breakout_v2
# Hypothesis: 4-hour Donchian breakout with 1-day trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 1.5x 20-period average volume.
# Short when: price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 1.5x 20-period average volume.
# Exit when: price crosses back through Donchian(20) middle (mean of high/low) OR opposite breakout occurs.
# Uses 1d for trend direction and 4h for entry/exit with volume confirmation to filter false breakouts.
# Target: 20-50 trades/year to minimize fee dust while capturing true breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_atr_breakout_v2"
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
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Calculate 20-period average volume
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_50 = np.zeros(len(close_1d))
    ema_1d_50[:] = np.nan
    ema_1d_50[49] = np.mean(close_1d[:50])
    for i in range(50, len(close_1d)):
        ema_1d_50[i] = close_1d[i] * 0.0377 + ema_1d_50[i-1] * 0.9623
    
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
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.5 * avg_vol if not np.isnan(avg_vol) else False
        
        if position == 1:  # Long
            # Exit: price below Donchian mid OR opposite breakout with volume
            if price < d_mid or (price < d_low and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short
            # Exit: price above Donchian mid OR opposite breakout with volume
            if price > d_mid or (price > d_high and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Entry: bullish breakout with volume and trend alignment
            if price > d_high and vol_surge and price > ema_1d:
                position = 1
                signals[i] = 0.30
            # Entry: bearish breakout with volume and trend alignment
            elif price < d_low and vol_surge and price < ema_1d:
                position = -1
                signals[i] = -0.30
    
    return signals