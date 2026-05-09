#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with volume confirmation and 12h EMA20 trend filter.
# Uses tighter breakout levels (R3/S3) for lower trade frequency to avoid overtrading.
# 12h EMA20 filters for trend direction to avoid counter-trend entries.
# Volume > 1.5x 20-period EMA ensures institutional participation.
# Designed to work in both bull and bear markets by following higher timeframe trend.
name = "4h_Camarilla_R3S3_Breakout_12hEMA20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA20 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: Range = High - Low
    range_1d = high_1d - low_1d
    r3 = close_1d + (range_1d * 1.1666)
    s3 = close_1d - (range_1d * 1.1666)
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    
    # Align to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3_shifted)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3_shifted)
    
    # 12h EMA20 trend filter
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and above 12h EMA20
            if (price > r3_4h[i] and vol_spike[i] and price > ema_20_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and below 12h EMA20
            elif (price < s3_4h[i] and vol_spike[i] and price < ema_20_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S3 (mean reversion to support)
            if price < s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R3 (mean reversion to resistance)
            if price > r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals