#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Williams Alligator (jaw/teeth/lips) identifies trending vs ranging markets via SMAs.
# In strong trends (Alligator "awake"), we trade breakouts in trend direction.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume confirmation (1.5x 20-period EMA) filters false breakouts.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h data
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(data, period):
        """Smoothed Moving Average ( Wilder's smoothing )"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[0:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + PRICE) / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Volume confirmation: 20-period EMA on 6h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start from 34 to have valid Alligator and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Alligator "awake" condition: lips, teeth, jaw are separated and ordered
        # For uptrend: lips > teeth > jaw
        # For downtrend: lips < teeth < jaw
        alligator_awake = (
            ((lips[i] > teeth[i]) and (teeth[i] > jaw[i])) or  # Uptrend alignment
            ((lips[i] < teeth[i]) and (teeth[i] < jaw[i]))     # Downtrend alignment
        )
        
        # Trend direction from 1d EMA34
        uptrend_1d = close > ema_34_1d_aligned[i]
        downtrend_1d = close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price > lips AND Alligator awake in uptrend alignment AND 1d uptrend AND volume spike
            if (close[i] > lips[i] and 
                lips[i] > teeth[i] and teeth[i] > jaw[i] and  # Lips > Teeth > Jaw (uptrend)
                uptrend_1d and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price < lips AND Alligator awake in downtrend alignment AND 1d downtrend AND volume spike
            elif (close[i] < lips[i] and 
                  lips[i] < teeth[i] and teeth[i] < jaw[i] and  # Lips < Teeth < Jaw (downtrend)
                  downtrend_1d and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < teeth OR Alligator starts sleeping (lips/teeth/jaw intertwined)
            if (close[i] < teeth[i] or 
                not ((lips[i] > teeth[i]) and (teeth[i] > jaw[i]))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > teeth OR Alligator starts sleeping (lips/teeth/jaw intertwined)
            if (close[i] > teeth[i] or 
                not ((lips[i] < teeth[i]) and (teeth[i] < jaw[i]))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals