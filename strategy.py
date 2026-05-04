#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) for trend identification,
# 1d EMA50 for higher timeframe trend filter (proven from top performers),
# and volume spike for confirmation. Designed for 12-37 trades/year to minimize fee drag.
# Works in bull markets via Alligator alignment (Jaw < Teeth < Lips) and in bear markets via reverse alignment.
# The 1d EMA50 provides a smooth trend filter that avoids whipsaw while capturing major moves.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_TrendFilter"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter from prior completed 1d bar
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_shifted = np.roll(ema50_1d, 1)
    ema50_1d_shifted[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_shifted)
    
    # Williams Alligator on 12h timeframe (using median price)
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator aligned bullish (Jaw < Teeth < Lips) AND price above 1d EMA50 AND volume spike
            if jaw[i] < teeth[i] < lips[i] and close[i] > ema50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator aligned bearish (Jaw > Teeth > Lips) AND price below 1d EMA50 AND volume spike
            elif jaw[i] > teeth[i] > lips[i] and close[i] < ema50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR price closes below 1d EMA50
            if not (jaw[i] < teeth[i] < lips[i]) or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR price closes above 1d EMA50
            if not (jaw[i] > teeth[i] > lips[i]) or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals