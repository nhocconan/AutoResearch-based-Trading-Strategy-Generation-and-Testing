#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation and 1d EMA50 trend filter.
- Long when price breaks above 1d H3 level + volume > 2.0x 6h volume MA(20) + price above 1d EMA50
- Short when price breaks below 1d L3 level + volume > 2.0x 6h volume MA(20) + price below 1d EMA50
- Fixed position size 0.25 to manage drawdown
- Uses Camarilla pivot structure (proven effective on 4h/12h) adapted to 6h timeframe with strict entry conditions
- Volume confirmation reduces false breakouts, EMA50 filter ensures alignment with daily trend
- Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for volume MA
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    
    # Volume average (20-period) on 6h for confirmation
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots and EMA50 trend filter (HTF for structure)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # Simplified: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Actually standard Camarilla: R3 = close + 1.1*(high-low)*1.1/4, R4 = close + 1.1*(high-low)*1.1/2
    # But we'll use H3/L3 as breakout levels (R3/S3)
    range_1d = high_1d - low_1d
    h3_1d = close_1d + 1.1 * range_1d * 1.1 / 4  # R3 level
    l3_1d = close_1d - 1.1 * range_1d * 1.1 / 4  # S3 level
    
    # Align all HTF indicators to primary timeframe (6h)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_20_6h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = volume_ma_20_aligned[i]
        ema_50_val = ema_50_aligned[i]
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for Camarilla H3/L3 breakouts with volume confirmation and 1d EMA50 trend filter
            # Long: price breaks above 1d H3 level + volume spike + price above 1d EMA50
            if price > h3 and vol > 2.0 * vol_ma and price > ema_50_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 1d L3 level + volume spike + price below 1d EMA50
            elif price < l3 and vol > 2.0 * vol_ma and price < ema_50_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 1d EMA50 (trend change) or opposite breakout level (L3)
            if price < ema_50_val or price < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above 1d EMA50 (trend change) or opposite breakout level (H3)
            if price > ema_50_val or price > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_VolumeSpike_1dEMA50"
timeframe = "6h"
leverage = 1.0