#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Camarilla H3/L3 breakout with volume confirmation and 1d EMA50 trend filter.
- Long when price breaks above 4h Camarilla H3 level + volume > 1.5x 20-period 1h volume MA + price above 1d EMA50
- Short when price breaks below 4h Camarilla L3 level + volume > 1.5x 20-period 1h volume MA + price below 1d EMA50
- Fixed position size 0.20 to manage drawdown in bear markets
- Uses proven edge: Camarilla levels (intraday support/resistance) + volume spike + HTF trend
- Designed for 1h timeframe with strict entry conditions to limit trades to 60-150 total over 4 years
- Session filter (08-20 UTC) to avoid low-liquidity periods
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
    
    # Get 4h data for Camarilla levels (HTF for structure)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate typical price for 4h
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    # Camarilla levels: H3/L3 = typical_price ± 1.1 * (high - low) / 2
    camarilla_h3_4h = typical_price_4h + 1.1 * (high_4h - low_4h) / 2.0
    camarilla_l3_4h = typical_price_4h - 1.1 * (high_4h - low_4h) / 2.0
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 1h for confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to primary timeframe (1h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma = volume_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        # Session filter: 08-20 UTC (intraday active hours)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for breakouts with volume confirmation and 1d EMA50 trend filter
            # Long: price breaks above Camarilla H3 + volume spike + price above 1d EMA50
            if price > camarilla_h3 and vol > 1.5 * vol_ma and price > ema_50_val:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla L3 + volume spike + price below 1d EMA50
            elif price < camarilla_l3 and vol > 1.5 * vol_ma and price < ema_50_val:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 1d EMA50 (trend change) or opposite Camarilla level
            if price < ema_50_val or price < camarilla_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit on close above 1d EMA50 (trend change) or opposite Camarilla level
            if price > ema_50_val or price > camarilla_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_VolumeSpike_1dEMA50_SessionFilter"
timeframe = "1h"
leverage = 1.0