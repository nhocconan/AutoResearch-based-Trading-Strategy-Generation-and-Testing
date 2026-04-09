#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_v2
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation and ATR filter.
# Long: Price breaks above R3 with volume > 1.5x 20-period average and ATR(14) > 0.01*close (volatility filter).
# Short: Price breaks below S3 with volume > 1.5x 20-period average and ATR(14) > 0.01*close.
# Exit: Price returns to pivot point (PP) for both long and short.
# Uses Camarilla pivots from 1d timeframe as structure levels.
# Volume and volatility filters reduce false breakouts. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r3 = pp + (range_1d * 3.0 / 8.0)
    
    # Support levels
    s3 = pp - (range_1d * 3.0 / 8.0)
    
    # Align Camarilla levels to 12h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Volatility filter: ATR > 1% of price (avoid low-volume choppy periods)
        vol_filter = atr[i] > 0.01 * close[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above R3 with volume and volatility confirmation
            if (close[i] > r3_aligned[i] and volume_confirmed and vol_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S3 with volume and volatility confirmation
            elif (close[i] < s3_aligned[i] and volume_confirmed and vol_filter):
                position = -1
                signals[i] = -0.25
    
    return signals