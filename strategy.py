#!/usr/bin/env python3
# 12h_1d_wkly_camarilla_volume_v1
# Hypothesis: 12-hour Camarilla pivot reversal with volume spike and weekly trend filter.
# Long: price touches S3 (1.1 level) AND volume > 2x 24-period average AND price > weekly EMA50.
# Short: price touches R3 (3.1 level) AND volume > 2x 24-period average AND price < weekly EMA50.
# Exit: price crosses H4/L4 levels or opposite Camarilla touch with volume.
# Designed for mean reversion in ranging markets and breakout in trending markets with strict entry to limit trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_wkly_camarilla_volume_v1"
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
    
    # Calculate typical price for pivot (typical price = (H+L+C)/3)
    typical_price = (high + low + close) / 3
    
    # 1-day pivot levels (using previous day's data)
    pivot = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's data for pivot calculation
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        pp = (ph + pl + pc) / 3
        
        pivot[i] = pp
        s1[i] = 2*pp - ph
        s2[i] = pp - (ph - pl)
        s3[i] = pl - 2*(ph - pl)
        r1[i] = 2*pp - pl
        r2[i] = pp + (ph - pl)
        r3[i] = ph + 2*(ph - pl)
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_50 = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_1w_50[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_1w_50[i] = close_1w[i] * (2/51) + ema_1w_50[i-1] * (49/51)
    
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # 24-period average volume (2 days of 12h data)
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        s3_level = s3[i]
        r3_level = r3[i]
        s4_level = s2[i]  # S2 acts as stop/reverse for longs
        r4_level = r2[i]  # R2 acts as stop/reverse for shorts
        ema_1w = ema_1w_50_aligned[i]
        
        if np.isnan(s3_level) or np.isnan(r3_level) or np.isnan(avg_vol) or np.isnan(ema_1w):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 2.0 * avg_vol
        
        if position == 1:  # Long position
            if price < s4_level or (price > r3_level and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price > r4_level or (price < s3_level and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: price touches S3 with volume surge and above weekly EMA
            if abs(price - s3_level) < 0.001 * price and vol_surge and price > ema_1w:
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 with volume surge and below weekly EMA
            elif abs(price - r3_level) < 0.001 * price and vol_surge and price < ema_1w:
                position = -1
                signals[i] = -0.25
    
    return signals