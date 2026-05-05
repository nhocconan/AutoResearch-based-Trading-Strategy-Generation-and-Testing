#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Camarilla pivot levels (R3/S3) from daily timeframe
# with volume spike confirmation and choppiness regime filter.
# Long when price breaks above R3 pivot level with volume > 2.0 * avg_volume(20) and CHOP > 61.8 (range)
# Short when price breaks below S3 pivot level with volume > 2.0 * avg_volume(20) and CHOP > 61.8 (range)
# Exit when price retouches the pivot point (PP) or volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla pivots provide statistically significant support/resistance levels
# Volume spike confirms breakout strength and reduces false signals
# Choppiness filter ensures we only trade in ranging markets where mean reversion works
# Works in bull markets (buying dips to S3 in uptrend via mean reversion) and bear markets (selling rallies to R3 in downtrend)

name = "12h_Camarilla_R3S3_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivots and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least one completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # R3 = Close + 1.1 * (High - Low) / 2
    # S3 = Close - 1.1 * (High - Low) / 2
    # PP = (High + Low + Close) / 3
    daily_range = high_1d - low_1d
    r3 = close_1d + 1.1 * daily_range / 2
    s3 = close_1d - 1.1 * daily_range / 2
    pp = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 12h timeframe (wait for completed daily bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate 12h choppiness index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * max(high-low))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # First bar
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values
    max_hl = pd.Series(high - low).rolling(window=14, min_periods=14).max().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr1 / (14 * max_hl)) / np.log10(14)
    chop_aligned = chop  # Already on 12h timeframe
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume confirmation and chop > 61.8 (ranging)
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                volume_confirm[i] and chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume confirmation and chop > 61.8 (ranging)
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                  volume_confirm[i] and chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retouches pivot point (PP) or volume drops below average
            if close[i] <= pp_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retouches pivot point (PP) or volume drops below average
            if close[i] >= pp_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals