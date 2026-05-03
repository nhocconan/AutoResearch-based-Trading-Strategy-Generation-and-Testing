#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA(34) trend filter + volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend absence (all lines intertwined) vs presence (diverged)
# In ranging markets (Alligator sleeping): fade extremes at 2x ATR from midpoint
# In trending markets (Alligator awakened): breakout continuation in direction of Jaw
# 1d EMA(34) ensures alignment with daily trend to avoid counter-trend trades
# Volume spike (>1.8x 20-period EMA) filters low-probability signals
# Designed for 6h timeframe to capture medium-term swings in both bull and bear markets
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator from 6h data (smoothed medians)
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods  
    # Lips: 5-period SMMA smoothed by 3 periods
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        sma = np.nansum(values[:period]) / period
        result[period-1] = sma
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    median_price = (high + low) / 2
    jaw = smma(smma(median_price, 13), 8)
    teeth = smma(smma(median_price, 8), 5)
    lips = smma(smma(median_price, 5), 3)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR for dynamic thresholds
    def atr(high, low, close, period=14):
        """Average True Range"""
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr_values = np.full(len(tr), np.nan)
        for i in range(period, len(tr)):
            if np.isnan(atr_values[i-1]):
                atr_values[i] = np.nanmean(tr[i-period+1:i+1])
            else:
                atr_values[i] = (atr_values[i-1] * (period-1) + tr[i]) / period
        return atr_values
    
    atr_14 = atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Alligator state: sleeping (intertwined) vs awakened (diverged)
        # Sleeping: max distance between any two lines < 0.5 * ATR
        max_jaw_teeth = np.abs(jaw[i] - teeth[i])
        max_teeth_lips = np.abs(teeth[i] - lips[i])
        max_jaw_lips = np.abs(jaw[i] - lips[i])
        max_distance = max(max_jaw_teeth, max_teeth_lips, max_jaw_lips)
        alligator_sleeping = max_distance < (0.5 * atr_14[i])
        alligator_awakened = not alligator_sleeping
        
        # Trend direction from 1d EMA(34)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            if alligator_sleeping:
                # Ranging market: fade extremes at 2x ATR from midpoint
                midpoint = (jaw[i] + teeth[i] + lips[i]) / 3
                upper_extreme = midpoint + (2.0 * atr_14[i])
                lower_extreme = midpoint - (2.0 * atr_14[i])
                
                if close[i] > upper_extreme and volume_spike and downtrend:
                    signals[i] = -0.25  # Short fade
                    position = -1
                elif close[i] < lower_extreme and volume_spike and uptrend:
                    signals[i] = 0.25   # Long fade
                    position = 1
            else:
                # Trending market: breakout continuation in direction of Jaw
                # Jaw slope determines trend direction
                if i >= 2:
                    jaw_slope = jaw[i] - jaw[i-2]
                    if jaw_slope > 0 and volume_spike and uptrend:
                        signals[i] = 0.25   # Long continuation
                        position = 1
                    elif jaw_slope < 0 and volume_spike and downtrend:
                        signals[i] = -0.25  # Short continuation
                        position = -1
        elif position == 1:
            # Exit long: Alligator starts sleeping OR price crosses below Teeth
            if alligator_sleeping or close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator starts sleeping OR price crosses above Teeth
            if alligator_sleeping or close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals