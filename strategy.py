#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w Camarilla H3/L3 breakout with volume confirmation and 1w EMA34 trend filter.
- Long when price breaks above 1w Camarilla H3 level + volume > 1.5x 20-period 1d volume MA + price above 1w EMA34
- Short when price breaks below 1w Camarilla L3 level + volume > 1.5x 20-period 1d volume MA + price below 1w EMA34
- Fixed position size 0.25 to manage drawdown
- Uses Camarilla H3/L3 (weekly levels) + volume spike + 1w EMA trend
- Designed for 1d timeframe with strict entry conditions to limit trades to 30-100 total over 4 years
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
    
    # Get 1d data for volume MA
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Volume average (20-period) on 1d for confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for Camarilla levels and EMA34 trend filter (HTF for structure)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate typical price for 1w
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    # Camarilla levels: H3/L3 = typical_price ± 1.1 * (high - low) / 4
    camarilla_h3_1w = typical_price_1w + 1.1 * (high_1w - low_1w) / 4.0
    camarilla_l3_1w = typical_price_1w - 1.1 * (high_1w - low_1w) / 4.0
    
    # Align all HTF indicators to primary timeframe (1d)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = volume_ma_20_aligned[i]
        ema_34_val = ema_34_aligned[i]
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 1w EMA34 trend filter
            # Long: price breaks above 1w Camarilla H3 + volume spike + price above 1w EMA34
            if price > camarilla_h3 and vol > 1.5 * vol_ma and price > ema_34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 1w Camarilla L3 + volume spike + price below 1w EMA34
            elif price < camarilla_l3 and vol > 1.5 * vol_ma and price < ema_34_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 1w EMA34 (trend change) or opposite Camarilla level
            if price < ema_34_val or price < camarilla_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above 1w EMA34 (trend change) or opposite Camarilla level
            if price > ema_34_val or price > camarilla_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_VolumeSpike_1wEMA34"
timeframe = "1d"
leverage = 1.0