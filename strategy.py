#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R3/S3) for breakout entries.
# Enter long when price breaks above R3 with volume > 2.0x 20-bar average and choppy market (CHOP > 61.8).
# Enter short when price breaks below S3 with volume > 2.0x 20-bar average and choppy market (CHOP > 61.8).
# Exit when price returns to the 1d Pivot Point (PP) level.
# Camarilla levels provide intraday support/resistance, volume confirms breakout strength,
# and choppy regime filter ensures we trade in ranging markets where mean reversion works.
# Uses discrete position sizing (0.25) to control risk. Target: 75-200 total trades over 4 years.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter_v2"
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
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    pp = typical_price  # Pivot Point
    
    # Calculate ranges
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r3 = pp + range_hl * 1.1 / 4.0  # Resistance 3
    s3 = pp - range_hl * 1.1 / 4.0  # Support 3
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate 4h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Calculate 4h Choppiness Index (CHOP) for regime filter
    # CHOP(14) = 100 * LOG10(SUM(ATR(1), 14) / (LOG10(MAX(HIGH,14) - MIN(LOW,14)))) / LOG10(14)
    # Simplified: CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # First bar TR
    atr1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # ATR(1) is just TR
    atr_sum_14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # Avoid division by zero
    chop = 100 * np.log10(atr_sum_14 / chop_denominator) / np.log10(14)
    chop_filter = chop > 61.8  # Choppy/ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(volume_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (close[i] > r3_aligned[i]) and volume_confirm[i] and chop_filter[i]
        short_entry = (close[i] < s3_aligned[i]) and volume_confirm[i] and chop_filter[i]
        
        # Exit condition: price returns to pivot point (PP)
        long_exit = close[i] <= pp_aligned[i]
        short_exit = close[i] >= pp_aligned[i]
        
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