#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND closes above 12h EMA34 AND volume > 1.5x average.
Short when price breaks below Camarilla S3 AND closes below 12h EMA34 AND volume > 1.5x average.
Exit when price reverts to Camarilla H3/L3 levels OR volume drops below average.
Uses 6h for price action, 12h for trend filter (HTF), and volume confirmation to reduce false breakouts.
Camarilla levels provide precise intraday support/resistance, EMA34 filters trend direction,
and volume ensures breakout conviction. Designed for 60-120 trades over 4 years (15-30/year).
Works in bull markets (breaks R3 with volume) and bear markets (breaks S3 with volume).
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
    
    # Get 6h data for Camarilla calculations
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Camarilla pivot levels for 6h timeframe
    # Pivot point = (high + low + close) / 3
    pp = (high_6h + low_6h + close_6h) / 3.0
    # Range = high - low
    rng = high_6h - low_6h
    
    # Camarilla levels
    # R4 = pp + (rng * 1.1/2)
    # R3 = pp + (rng * 1.1/4)
    # R2 = pp + (rng * 1.1/6)
    # R1 = pp + (rng * 1.1/12)
    # S1 = pp - (rng * 1.1/12)
    # S2 = pp - (rng * 1.1/6)
    # S3 = pp - (rng * 1.1/4)
    # S4 = pp - (rng * 1.1/2)
    camarilla_pp = pp
    camarilla_r3 = camarilla_pp + (rng * 1.1 / 4)
    camarilla_s3 = camarilla_pp - (rng * 1.1 / 4)
    camarilla_h3 = camarilla_pp + (rng * 1.1 / 12)  # H3 = R1
    camarilla_l3 = camarilla_pp - (rng * 1.1 / 12)  # L3 = S1
    
    # Calculate volume average (20-period) on 6h
    volume_series = pd.Series(volume_6h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_34 = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 6h Camarilla levels, volume MA, and 12h EMA34 to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l3)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema_val = ema_34_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R3 AND close > 12h EMA34 AND volume > 1.5x avg
            if price > r3 and price > ema_val and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S3 AND close < 12h EMA34 AND volume > 1.5x avg
            elif price < s3 and price < ema_val and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla H3 OR volume < average
            if price < h3 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla L3 OR volume < average
            if price > l3 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Volume_12hEMA34_Filter"
timeframe = "6h"
leverage = 1.0