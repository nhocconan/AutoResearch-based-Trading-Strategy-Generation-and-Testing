#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with volume spike and 12h EMA34 trend filter.
Long when price breaks above Camarilla R3 level AND volume > 1.5x 20-period average AND price > 12h EMA34.
Short when price breaks below Camarilla S3 level AND volume > 1.5x 20-period average AND price < 12h EMA34.
Exit when price reverts to Camarilla Pivot point (PP).
Uses 6h for price/volume/Camarilla levels, 12h for EMA34 trend filter to avoid whipsaw.
Camarilla pivots provide statistically significant support/resistance levels, volume reduces fakeouts,
12h EMA ensures we trade with intermediate trend. Works in bull/bear markets by adapting to trend.
Targets 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla calculations and volume
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Camarilla pivot levels on 6h timeframe (based on previous bar)
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    # We use the previous completed 6h bar's HLC to calculate levels for current bar
    pp = (np.roll(high_6h, 1) + np.roll(low_6h, 1) + np.roll(close_6h, 1)) / 3
    r3 = pp + (np.roll(high_6h, 1) - np.roll(low_6h, 1)) * 1.1 / 2
    s3 = pp - (np.roll(high_6h, 1) - np.roll(low_6h, 1)) * 1.1 / 2
    # Pivot point (PP) for exit
    pivot_pp = pp
    
    # Calculate volume average (20-period) on 6h
    volume_series = pd.Series(volume_6h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 6h indicators to 6h timeframe (no additional delay needed as we used previous bar)
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3)
    pivot_pp_aligned = align_htf_to_ltf(prices, df_6h, pivot_pp)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pivot_pp_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        pivot_level = pivot_pp_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema_34 = ema_34_12h_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R3 AND volume > 1.5x avg AND price > 12h EMA34 (bullish trend)
            if price > r3_level and vol > 1.5 * vol_ma and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S3 AND volume > 1.5x avg AND price < 12h EMA34 (bearish trend)
            elif price < s3_level and vol > 1.5 * vol_ma and price < ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla Pivot Point
            if price < pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla Pivot Point
            if price > pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_CamarillaR3S3_Volume_12hEMA34_Filter"
timeframe = "6h"
leverage = 1.0