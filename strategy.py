#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA(34) trend filter and volume confirmation
# Williams Alligator (JAW/TEETH/LIPS) identifies trending vs ranging markets
# Entry: Alligator lines aligned (JAW > TEETH > LIPS for long, reverse for short) + price outside lips
# 1d EMA(34) ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume spike (>2.0x 20-period EMA) filters low-probability breakouts
# Works in bull/bear markets by following 1d trend direction for entries
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag

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
    
    # Williams Alligator: SMAs of median price with specific periods
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # JAW: 13-period SMMA, shifted 8 bars ahead
    # TEETH: 8-period SMMA, shifted 5 bars ahead  
    # LIPS: 5-period SMMA, shifted 3 bars ahead
    # Using SMA as approximation for SMMA (simple moving average)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams Alligator signals with 1d trend filter
        # Long: JAW > TEETH > LIPS (bullish alignment) AND price > LIPS AND above 1d EMA34
        # Short: JAW < TEETH < LIPS (bearish alignment) AND price < LIPS AND below 1d EMA34
        if position == 0:
            if jaw_vals[i] > teeth_vals[i] and teeth_vals[i] > lips_vals[i] and close[i] > lips_vals[i] and close[i] > ema_34_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif jaw_vals[i] < teeth_vals[i] and teeth_vals[i] < lips_vals[i] and close[i] < lips_vals[i] and close[i] < ema_34_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines reverse alignment OR price crosses below LIPS OR below 1d EMA34
            if (jaw_vals[i] <= teeth_vals[i] or teeth_vals[i] <= lips_vals[i] or 
                close[i] <= lips_vals[i] or close[i] <= ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines reverse alignment OR price crosses above LIPS OR above 1d EMA34
            if (jaw_vals[i] >= teeth_vals[i] or teeth_vals[i] >= lips_vals[i] or 
                close[i] >= lips_vals[i] or close[i] >= ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals