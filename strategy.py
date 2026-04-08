#!/usr/bin/env python3
# 12h_1d_wkly_camarilla_volume_v2
# Hypothesis: 12-hour Camarilla pivot reversal with volume confirmation and weekly trend filter.
# Long: price touches Camarilla S3 (support) from above, volume > 1.5x avg, price > weekly EMA50.
# Short: price touches Camarilla R3 (resistance) from below, volume > 1.5x avg, price < weekly EMA50.
# Exit: price crosses Camarilla pivot point (center) or opposite S3/R3 touch with volume.
# Designed for mean-reversion in ranging markets with trend filter to avoid counter-trend trades.
# Weekly EMA filter ensures alignment with higher timeframe trend, reducing whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_wkly_camarilla_volume_v2"
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
    
    # Calculate 12h Camarilla levels (using previous period's HLC)
    camarilla_pivot = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's HLC for current bar's levels (no look-ahead)
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        pivot = (ph + pl + pc) / 3
        range_ = ph - pl
        
        camarilla_pivot[i] = pivot
        camarilla_s3[i] = pc - 1.1 * range_  # S3 level
        camarilla_r3[i] = pc + 1.1 * range_  # R3 level
    
    # 20-period average volume for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        pivot = camarilla_pivot[i]
        s3 = camarilla_s3[i]
        r3 = camarilla_r3[i]
        ema_1w = ema_1w_50_aligned[i]
        
        if np.isnan(pivot) or np.isnan(s3) or np.isnan(r3) or np.isnan(avg_vol) or np.isnan(ema_1w):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.5 * avg_vol
        
        if position == 1:  # Long position
            if price > pivot or (price < s3 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price < pivot or (price > r3 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: price touches S3 from above with volume surge and above weekly EMA
            if price <= s3 and close[i-1] > s3 and vol_surge and price > ema_1w:
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 from below with volume surge and below weekly EMA
            elif price >= r3 and close[i-1] < r3 and vol_surge and price < ema_1w:
                position = -1
                signals[i] = -0.25
    
    return signals