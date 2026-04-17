#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla H3/L3 breakout with volume confirmation and 4h EMA50 trend filter.
- Long when price breaks above 1d Camarilla H3 level + volume > 2.0x 20-period 4h volume MA + price above 4h EMA50
- Short when price breaks below 1d Camarilla L3 level + volume > 2.0x 20-period 4h volume MA + price below 4h EMA50
- Fixed position size 0.25 to manage drawdown
- Uses Camarilla H3/L3 (stronger daily levels) + volume spike + 4h EMA trend
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
- Works in bull markets (buying breakouts with uptrend) and bear markets (selling breakdowns with downtrend)
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
    
    # Get 4h data for EMA50 trend filter and volume MA
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 4h for confirmation
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla levels (HTF for structure)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate typical price for 1d
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels: H3/L3 = typical_price ± 1.1 * (high - low) / 4
    camarilla_h3_1d = typical_price_1d + 1.1 * (high_1d - low_1d) / 4.0
    camarilla_l3_1d = typical_price_1d - 1.1 * (high_1d - low_1d) / 4.0
    
    # Align all HTF indicators to primary timeframe (4h)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50_val = ema_50_aligned[i]
        vol_ma = volume_ma_20_aligned[i]
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 4h EMA50 trend filter
            # Long: price breaks above 1d Camarilla H3 + volume spike + price above 4h EMA50
            if price > camarilla_h3 and vol > 2.0 * vol_ma and price > ema_50_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 1d Camarilla L3 + volume spike + price below 4h EMA50
            elif price < camarilla_l3 and vol > 2.0 * vol_ma and price < ema_50_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 4h EMA50 (trend change) or opposite Camarilla level
            if price < ema_50_val or price < camarilla_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above 4h EMA50 (trend change) or opposite Camarilla level
            if price > ema_50_val or price > camarilla_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_VolumeSpike_4hEMA50"
timeframe = "4h"
leverage = 1.0