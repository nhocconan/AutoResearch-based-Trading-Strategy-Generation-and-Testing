#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via alignment of smoothed medians
# EMA50 on 1d provides higher timeframe trend direction to avoid counter-trend trades
# Volume confirmation (>1.7x 20 EMA) ensures breakout participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# Works in both bull and bear: Alligator catches trends, EMA50 filter avoids whipsaws in ranging markets.

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 4h timeframe
    # Jaw: 13-period smoothed median, Teeth: 8-period, Lips: 5-period
    median_price = (high + low) / 2.0
    
    # Jaw (13)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).apply(
        lambda x: np.median(x), raw=True
    ).values
    jaw = pd.Series(jaw).ewm(span=5, adjust=False, min_periods=5).mean().values  # smoothed
    
    # Teeth (8)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).apply(
        lambda x: np.median(x), raw=True
    ).values
    teeth = pd.Series(teeth).ewm(span=3, adjust=False, min_periods=3).mean().values  # smoothed
    
    # Lips (5)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).apply(
        lambda x: np.median(x), raw=True
    ).values
    lips = pd.Series(lips).ewm(span=2, adjust=False, min_periods=2).mean().values  # smoothed
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.7 x 20-period EMA
        volume_confirm = volume[i] > (1.7 * vol_ema_20[i])
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_gt_teeth = lips[i] > teeth[i]
        teeth_gt_jaw = teeth[i] > jaw[i]
        lips_lt_teeth = lips[i] < teeth[i]
        teeth_lt_jaw = teeth[i] < jaw[i]
        
        if position == 0:
            # Long conditions: Alligator aligned up + price above Lips + uptrend + volume spike
            if lips_gt_teeth and teeth_gt_jaw and close[i] > lips[i] and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator aligned down + price below Lips + downtrend + volume spike
            elif lips_lt_teeth and teeth_lt_jaw and close[i] < lips[i] and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks down OR price returns to Teeth OR weak volume
            if (not (lips_gt_teeth and teeth_gt_jaw) or 
                close[i] < teeth[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks up OR price returns to Teeth OR weak volume
            if (not (lips_lt_teeth and teeth_lt_jaw) or 
                close[i] > teeth[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals