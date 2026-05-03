#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via alignment of smoothed medians.
# In bull market: Lips > Teeth > Jaw (all rising) = uptrend
# In bear market: Lips < Teeth < Jaw (all falling) = downtrend
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Volume confirmation filters false signals. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via long entries on bullish Alligator alignment and in bear markets via short entries on bearish alignment.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Smoothed medians (using close as proxy for median price)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    median_price = (high + low + close) / 3.0
    
    # Jaw (13, 8)
    jaw = pd.Series(median_price).ewm(alpha=1/13, adjust=False).mean().shift(8).values
    # Teeth (8, 5)
    teeth = pd.Series(median_price).ewm(alpha=1/8, adjust=False).mean().shift(5).values
    # Lips (5, 3)
    lips = pd.Series(median_price).ewm(alpha=1/5, adjust=False).mean().shift(3).values
    
    # Volume confirmation: 20-period EMA on 6h
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start from 13 to have valid Alligator values
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Bullish alignment: Lips > Teeth > Jaw (all rising)
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        # Bearish alignment: Lips < Teeth < Jaw (all falling)
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        if position == 0:
            # Long: bullish Alligator alignment with daily uptrend and volume spike
            if bullish_alignment and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment with daily downtrend and volume spike
            elif bearish_alignment and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: loss of bullish alignment or daily trend reversal
            if not bullish_alignment or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: loss of bearish alignment or daily trend reversal
            if not bearish_alignment or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals