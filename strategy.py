#!/usr/bin/env python3
"""
12h_WilliamsAlligator_Strategy_v1
Hypothesis: Uses Williams Alligator (Jaw, Teeth, Lips) on 12h timeframe with
1d trend filter and volume confirmation. The Alligator identifies trends when
its lines are separated and aligned. In bull markets: Jaw (13-period) below
Teeth (8-period) below Lips (5-period). In bear markets: reverse.
Trades only when Alligator is "awake" (lines separated) and price confirms
with volume spike. Avoids whipsaw by requiring 1d EMA34 trend alignment.
Designed for low trade frequency (<30/year) to minimize fee drag.
Works in both bull and bear markets by following the dominant trend.
"""

name = "12h_WilliamsAlligator_Strategy_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 12h: Smoothed Median Price (HLC/3)
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    
    # Alligator lines: Smoothed with 5,8,13 periods, shifted 3,5,8 bars
    jaw = tp_series.rolling(window=13, center=False).mean().shift(8)   # Blue line (13-period)
    teeth = tp_series.rolling(window=8, center=False).mean().shift(5)   # Red line (8-period)
    lips = tp_series.rolling(window=5, center=False).mean().shift(3)    # Green line (5-period)
    
    # Convert to numpy arrays
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Aligator is "awake" when lines are separated (not intertwined)
    # Bullish: Lips > Teeth > Jaw
    # Bearish: Jaw > Teeth > Lips
    bullish_aligned = (lips > teeth) & (teeth > jaw)
    bearish_aligned = (jaw > teeth) & (teeth > lips)
    
    # Align to 12h timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_aligned.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_aligned.astype(float))
    
    # Volume confirmation: 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    
    # 1d trend filter: EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Alligator (need 13+8=21 bars)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Bullish Alligator alignment + price above 1d EMA + volume spike
            if (bullish_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + price below 1d EMA + volume spike
            elif (bearish_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when Alligator goes to sleep (lines intertwine) or trend fails
            if position == 1:
                # Exit long: Alligator not bullish OR price below 1d EMA
                if not bullish_aligned[i] or close[i] < ema_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Alligator not bearish OR price above 1d EMA
                if not bearish_aligned[i] or close[i] > ema_34_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals