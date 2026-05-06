#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Uses 12h EMA50 for stronger trend alignment (reduces whipsaw vs 1d EMA34)
# Volume spike (>1.8x 20-bar average) confirms breakout strength with higher threshold
# ATR-based stoploss via signal=0 when price retests opposite Camarilla level
# Discrete sizing 0.25 to limit fee drag; target 75-180 total trades over 4 years (19-45/year)
# Proven pattern: price channel breakouts with volume confirmation work on BTC/ETH in both bull/bear markets
# Using 12h HTF (per experiment instructions) for better trend stability

name = "4h_Camarilla_R3S3_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop - using 12h as specified
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA50 trend filter (stronger trend filter)
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume spike filter (>1.8x 20-bar average for stricter confirmation)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Calculate 4h Camarilla pivot levels (using previous 12h bar)
    # Camarilla: R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
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
    
    # Align HTF indicators to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    volume_filter_aligned = align_htf_to_ltf(prices, df_12h, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend (price > EMA50) AND volume spike
            if close[i] > camarilla_high_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 AND downtrend (price < EMA50) AND volume spike
            elif close[i] < camarilla_low_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests S3 from above (trend reversal)
            if close[i] <= camarilla_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests R3 from below (trend reversal)
            if close[i] >= camarilla_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals