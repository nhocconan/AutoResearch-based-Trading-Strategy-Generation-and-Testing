#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R3/S3) breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above R3 (1d) AND 1d volume > 2x 20-period MA AND 1d chop > 61.8 (range) → mean reversion fade.
# Short when price breaks below S3 (1d) AND 1d volume > 2x 20-period MA AND 1d chop > 61.8 (range) → mean reversion fade.
# Exit when price reverts to 1d EMA34 OR chop < 38.2 (trend) OR volume drops below 1.5x MA.
# Uses 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Camarilla pivots identify key reversal levels, volume confirms participation, chop filter ensures ranging markets for mean reversion.
# Designed to work in ranging markets (chop > 61.8) which frequently occur in BTC/ETH consolidation phases.

name = "12h_Camarilla_R3S3_VolumeSpike_Chop_MeanRev"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivots, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3, EMA34)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels
    r3_1d = pp_1d + range_1d * 1.1 / 4  # R3 = PP + (H-L)*1.1/4
    s3_1d = pp_1d - range_1d * 1.1 / 4  # S3 = PP - (H-L)*1.1/4
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP) - range detector
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # ATR(14) - Wilder's smoothing
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # CHOP = 100 * log10(sum_TR_14 / (HH_14 - LL_14)) / log10(14)
    # Avoid division by zero and log of zero
    hh_ll_diff = hh_14 - ll_14
    chop_1d = np.full_like(close_1d, np.nan)
    valid = (hh_ll_diff > 0) & (~np.isnan(sum_tr_14))
    chop_1d[valid] = 100 * np.log10(sum_tr_14[valid] / hh_ll_diff[valid]) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 1d volume > 2x 20-period volume MA
        # Note: We need current 1d volume, but we're in 12h timeframe
        # Use the volume from the completed 1d bar that corresponds to this 12h bar
        volume_spike = volume_1d[i//2] > (volume_ma_20_1d_aligned[i] * 2.0) if i//2 < len(volume_1d) else False
        
        # Choppiness regime: CHOP > 61.8 = ranging (good for mean reversion)
        chop_ranging = chop_1d_aligned[i] > 61.8
        chop_trending = chop_1d_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above R3 AND volume spike AND ranging market
            if close[i] > r3_1d_aligned[i] and volume_spike and chop_ranging:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND volume spike AND ranging market
            elif close[i] < s3_1d_aligned[i] and volume_spike and chop_ranging:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to EMA34 OR chop becomes trending OR volume drops
            if close[i] < ema34_1d_aligned[i] or chop_trending or (volume_1d[i//2] < volume_ma_20_1d_aligned[i] * 1.5 if i//2 < len(volume_1d) else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to EMA34 OR chop becomes trending OR volume drops
            if close[i] > ema34_1d_aligned[i] or chop_trending or (volume_1d[i//2] < volume_ma_20_1d_aligned[i] * 1.5 if i//2 < len(volume_1d) else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals