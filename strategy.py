#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 12h EMA34 is rising AND volume > 1.3x average.
Short when price breaks below Camarilla S3 AND 12h EMA34 is falling AND volume > 1.3x average.
Exit when price reverts to Camarilla pivot point (PP) OR EMA34 flips direction.
Uses 6h for price/volume and 12h for EMA filter to reduce whipsaw and capture medium-term trends.
Camarilla levels provide precise intraday support/resistance; EMA34 filters choppy markets.
Target: 50-150 total trades over 4 years (12-37/year). Works in bull markets (breaks R3/S3 with trend)
and bear markets (fades at R3/S3 during rallies, breaks down in downtrends).
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
    
    # Get 6h data for Camarilla calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Camarilla levels on 6h timeframe (based on previous bar)
    # Camarilla: PP = (H+L+C)/3, R3 = PP + (H-L)*1.1/2, S3 = PP - (H-L)*1.1/2
    typical_price = (high_6h + low_6h + close_6h) / 3.0
    pp_6h = typical_price
    range_6h = high_6h - low_6h
    r3_6h = pp_6h + (range_6h * 1.1 / 2.0)
    s3_6h = pp_6h - (range_6h * 1.1 / 2.0)
    
    # Get 12h data for EMA34 filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 6h Camarilla and volume to 6h timeframe (no alignment needed for same TF)
    pp_aligned = pp_6h
    r3_aligned = r3_6h
    s3_aligned = s3_6h
    
    # Align 12h EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume average (20-period) on 6h
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        pp = pp_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # EMA direction: rising if current > previous, falling if current < previous
        if i > start_idx:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: price > R3 AND EMA rising AND volume > 1.3x avg
            if price > r3 and ema_rising and vol > 1.3 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < S3 AND EMA falling AND volume > 1.3x avg
            elif price < s3 and ema_falling and vol > 1.3 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < PP OR EMA falls
            if price < pp or not ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > PP OR EMA rises
            if price > pp or not ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Volume_EMA34_Filter"
timeframe = "6h"
leverage = 1.0