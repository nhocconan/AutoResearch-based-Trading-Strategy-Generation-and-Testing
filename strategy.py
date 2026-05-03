#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period MA AND 1d chop > 61.8 (range).
# Short when price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period MA AND 1d chop > 61.8 (range).
# Exit when price reverts to Camarilla Pivot level OR chop < 38.2 (trend regime).
# Uses 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Camarilla provides precise intraday support/resistance, volume confirms participation, chop filter avoids trending markets where mean reversion fails.

name = "12h_Camarilla_R3S3_VolumeSpike_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels use previous day's range
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    range_1d = prev_high - prev_low
    
    # Camarilla levels
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = camarilla_pivot + (range_1d * 1.1 / 4)
    camarilla_s3 = camarilla_pivot - (range_1d * 1.1 / 4)
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d Choppiness Index (CHOP)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # ATR(14) - smoothed TR
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # CHOP = 100 * log10(sum TR(14) / (ATR(14) * 14)) / log10(14)
    chop_1d = 100 * np.log10(sum_tr_14 / (atr_1d * 14)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 1d volume > 2.0x 20-period volume MA
        volume_spike = volume_1d[i] > (volume_ma_20_1d[i] * 2.0)
        
        # Chop regime condition: CHOP > 61.8 = ranging (good for mean reversion)
        chop_ranging = chop_1d_aligned[i] > 61.8
        chop_trending = chop_1d_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above R3 AND volume spike AND chop ranging AND session
            if close[i] > camarilla_r3_aligned[i] and volume_spike and chop_ranging:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND volume spike AND chop ranging AND session
            elif close[i] < camarilla_s3_aligned[i] and volume_spike and chop_ranging:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot OR chop becomes trending
            if close[i] <= camarilla_pivot_aligned[i] or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot OR chop becomes trending
            if close[i] >= camarilla_pivot_aligned[i] or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals