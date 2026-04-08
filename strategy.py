#!/usr/bin/env python3
# 12h_1w_donchian_volume_breakout_v1
# Hypothesis: 12-hour Donchian breakout with 1-week trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) high AND price > 1w EMA100 AND volume > 2.0x 20-period average volume.
# Short when: price breaks below Donchian(20) low AND price < 1w EMA100 AND volume > 2.0x 20-period average volume.
# Exit when: price crosses back through Donchian(20) middle (mean of high/low) OR opposite breakout occurs.
# Uses 1w for trend direction and 12h for entry/exit with volume confirmation to filter false breakouts.
# Target: 15-35 trades/year to minimize fee dust while capturing true breakouts.

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
    
    # Calculate 12h Donchian channels (20-period)
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
    
    # Get 1w EMA100 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_100 = np.zeros(len(close_1w))
    ema_1w_100[:] = np.nan
    ema_1w_100[99] = np.mean(close_1w[:100])
    for i in range(100, len(close_1w)):
        ema_1w_100[i] = close_1w[i] * 0.0196 + ema_1w_100[i-1] * 0.9804
    
    ema_1w_100_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_100)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        d_mid = donchian_mid[i]
        ema_1w = ema_1w_100_aligned[i]
        
        if np.isnan(d_high) or np.isnan(d_low) or np.isnan(d_mid) or np.isnan(avg_vol) or np.isnan(ema_1w):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 2.0 * avg_vol if not np.isnan(avg_vol) else False
        
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
            if price > d_high and vol_surge and price > ema_1w:
                position = 1
                signals[i] = 0.30
            # Entry: bearish breakout with volume and trend alignment
            elif price < d_low and vol_surge and price < ema_1w:
                position = -1
                signals[i] = -0.30
    
    return signals