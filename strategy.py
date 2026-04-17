#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 12h EMA34 is rising AND volume > 1.3x average.
Short when price breaks below Camarilla S3 AND 12h EMA34 is falling AND volume > 1.3x average.
Exit when price reverts to Camarilla R1/S1 or 12h EMA34 flips direction.
Uses 6h for entry timing and 12h for trend filter to avoid whipsaw in ranging markets.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla levels provide institutional
reference points, EMA filter ensures trend alignment, volume confirms breakout strength.
Works in bull markets (breaks above R3 with rising EMA) and bear markets (breaks below S3 with falling EMA).
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
    
    # Calculate Camarilla levels on 6h timeframe (based on previous bar's range)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    #          R1 = close + 1.1*(high-low)*1.1/6, S1 = close - 1.1*(high-low)*1.1/6
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    prev_high[0] = high_6h[0]  # first bar
    prev_low[0] = low_6h[0]
    prev_close[0] = close_6h[0]
    
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * rang * 1.1 / 4
    camarilla_s3 = prev_close - 1.1 * rang * 1.1 / 4
    camarilla_r1 = prev_close + 1.1 * rang * 1.1 / 6
    camarilla_s1 = prev_close - 1.1 * rang * 1.1 / 6
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_rising = ema_12h > np.roll(ema_12h, 1)  # rising if current > previous
    ema_12h_falling = ema_12h < np.roll(ema_12h, 1)  # falling if current < previous
    
    # Volume average (20-period) on 6h
    volume_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s1)
    ema_12h_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_rising)
    ema_12h_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_falling)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_12h_rising_aligned[i]) or np.isnan(ema_12h_falling_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema_rising = bool(ema_12h_rising_aligned[i])
        ema_falling = bool(ema_12h_falling_aligned[i])
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R3 AND 12h EMA34 rising AND volume > 1.3x average
            if price > r3 and ema_rising and vol > 1.3 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S3 AND 12h EMA34 falling AND volume > 1.3x average
            elif price < s3 and ema_falling and vol > 1.3 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla R1 OR 12h EMA34 falling
            if price < r1 or not ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla S1 OR 12h EMA34 rising
            if price > s1 or not ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Volume_12hEMA34_Filter"
timeframe = "6h"
leverage = 1.0