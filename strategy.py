#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d VWAP trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels: %R > -20 = overbought, %R < -80 = oversold.
# Uses 1d VWAP for trend filter to align with institutional money flow.
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
# Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 1d data for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d VWAP
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: price below VWAP (downtrend) + oversold + volume
        if close[i] < vwap_1d_aligned[i] and williams_r[i] < -80 and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short setup: price above VWAP (uptrend) + overbought + volume
        elif close[i] > vwap_1d_aligned[i] and williams_r[i] > -20 and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit conditions
        elif position == 1 and (williams_r[i] > -50 or close[i] > vwap_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (williams_r[i] < -50 or close[i] < vwap_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_1dVWAP_VolumeFilter"
timeframe = "4h"
leverage = 1.0