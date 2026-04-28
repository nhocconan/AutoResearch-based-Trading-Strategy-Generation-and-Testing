#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels (R3/S3) with volume spike confirmation and choppiness regime filter.
# Enter long when price breaks above R3 with volume > 2.0x 20-bar average and CHOP > 61.8 (ranging market).
# Enter short when price breaks below S3 with volume > 2.0x 20-bar average and CHOP > 61.8.
# Exit when price re-enters the Camarilla range (between H3 and L3) or opposite pivot break occurs.
# Camarilla levels provide precise support/resistance, volume confirms breakout strength, chop filter avoids false breakouts in trends.
# Works in bull markets (buying breakouts above R3) and bear markets (selling breakdowns below S3).
# Uses discrete position sizing (0.25) to control risk. Target: 75-200 total trades over 4 years.

name = "4h_Camarilla_R3S3_Breakout_12hVolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (based on previous day's high, low, close)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_12h + low_12h + close_12h) / 3.0
    
    # R3 = Close + (High - Low) * 1.1
    r3 = close_12h + (high_12h - low_12h) * 1.1
    
    # S3 = Close - (High - Low) * 1.1
    s3 = close_12h - (high_12h - low_12h) * 1.1
    
    # H3 = High + 2 * (Close - Low) * 1.1 / 2
    h3 = high_12h + 2 * (close_12h - low_12h) * 1.1 / 2
    
    # L3 = Low - 2 * (High - Close) * 1.1 / 2
    l3 = low_12h - 2 * (high_12h - close_12h) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    
    # Calculate 4h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Calculate 4h choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = highest_high_14 - lowest_low_14
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    chop = 100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / 
                           (np.log10(14) * denominator))
    
    # CHOP > 61.8 = ranging market (good for mean reversion/breakouts in range)
    chop_ranging = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_ma_20[i]) or np.isnan(chop_ranging[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (close[i] > r3_aligned[i]) and volume_confirm[i] and chop_ranging[i]
        short_entry = (close[i] < s3_aligned[i]) and volume_confirm[i] and chop_ranging[i]
        
        # Exit conditions: price re-enters Camarilla range (between H3 and L3) or opposite pivot break
        long_exit = (close[i] < h3_aligned[i]) or (close[i] > r3_aligned[i] and not volume_confirm[i])  # price back below H3 or volume fails
        short_exit = (close[i] > l3_aligned[i]) or (close[i] < s3_aligned[i] and not volume_confirm[i])  # price back above L3 or volume fails
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals