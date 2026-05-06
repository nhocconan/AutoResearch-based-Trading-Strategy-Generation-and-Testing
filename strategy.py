#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 12h Camarilla R3/S3 breakout with volume confirmation
# Williams Alligator (jaw/teeth/lips) identifies trendless markets - only trade when aligned (trending)
# 12h Camarilla R3/S3 provides institutional support/resistance levels for breakouts
# Volume spike (>2.0x 50-bar average) confirms institutional participation
# Discrete sizing 0.25 to balance return and drawdown; target 80-120 total trades over 4 years
# Works in bull/bear: Alligator filters sideways markets, Camarilla breakouts capture institutional moves

name = "6h_WilliamsAlligator_12hCamarilla_R3S3_Breakout_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Williams Alligator on 12h: Jaw(13,8), Teeth(8,5), Lips(5,3) - SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Alligator alignment: all three lines in order (trending market)
    # Bullish: Lips > Teeth > Jaw
    # Bearish: Lips < Teeth < Jaw
    alligator_bullish = (lips > teeth) & (teeth > jaw)
    alligator_bearish = (lips < teeth) & (teeth < jaw)
    
    # Calculate 12h Camarilla pivot levels (using previous 12h bar)
    camarilla_high = []
    camarilla_low = []
    for i in range(len(close_12h)):
        if i == 0:
            camarilla_high.append(np.nan)
            camarilla_low.append(np.nan)
        else:
            h = high_12h[i-1]
            l = low_12h[i-1]
            c = close_12h[i-1]
            r3 = c + ((h - l) * 1.1 / 4)
            s3 = c - ((h - l) * 1.1 / 4)
            camarilla_high.append(r3)
            camarilla_low.append(s3)
    
    camarilla_high = np.array(camarilla_high)
    camarilla_low = np.array(camarilla_low)
    
    # Volume spike filter (>2.0x 50-bar average on 12h)
    vol_ma_50 = pd.Series(volume_12h).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume_12h > (2.0 * vol_ma_50)
    
    # Align HTF indicators to 6h timeframe
    alligator_bullish_aligned = align_htf_to_ltf(prices, df_12h, alligator_bullish.astype(float))
    alligator_bearish_aligned = align_htf_to_ltf(prices, df_12h, alligator_bearish.astype(float))
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    volume_filter_aligned = align_htf_to_ltf(prices, df_12h, volume_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(alligator_bullish_aligned[i]) or np.isnan(alligator_bearish_aligned[i]) or 
            np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish_alligator = bool(alligator_bullish_aligned[i])
        bearish_alligator = bool(alligator_bearish_aligned[i])
        volume_spike = bool(volume_filter_aligned[i])
        
        if position == 0:
            # Long breakout: price > R3 AND bullish alligator AND volume spike
            if close[i] > camarilla_high_aligned[i] and bullish_alligator and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 AND bearish alligator AND volume spike
            elif close[i] < camarilla_low_aligned[i] and bearish_alligator and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests S3 from above OR alligator turns bearish
            if close[i] <= camarilla_low_aligned[i] or not bullish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests R3 from below OR alligator turns bullish
            if close[i] >= camarilla_high_aligned[i] or not bearish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals