#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend presence via convergence/divergence.
# In trending markets (JAW > TEETH > LIPS for down, reverse for up), trade with 1d trend.
# In ranging markets (alligator sleeping: lines intertwined), avoid trading.
# Volume filter ensures breakouts have conviction. Works in bull/bear via 1d trend filter.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Williams Alligator lines (all based on SMAs of median price)
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMA, shifted 8 bars ahead
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: 8-period SMA, shifted 5 bars ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: 5-period SMA, shifted 3 bars ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Align Williams Alligator to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # Same timeframe, just shift
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # 1d EMA trend filter (34-period)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.5 x 24-period average (4 days of 6h bars)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator (max shift 8), EMA (34), volume MA (24)
    start_idx = max(8, 34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_34_aligned[i]
        bearish_trend = price < ema_34_aligned[i]
        
        # Alligator conditions
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Alligator awake and trending up: Lips > Teeth > Jaw
        alligator_up = lips_val > teeth_val and teeth_val > jaw_val
        # Alligator awake and trending down: Jaw > Teeth > Lips
        alligator_down = jaw_val > teeth_val and teeth_val > lips_val
        # Alligator sleeping (no trend): lines intertwined
        alligator_sleeping = not (alligator_up or alligator_down)
        
        if position == 0:
            # Long: alligator up + volume + bullish 1d trend
            if alligator_up and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: alligator down + volume + bearish 1d trend
            elif alligator_down and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: alligator turns down or trend turns bearish
            if not alligator_up or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: alligator turns up or trend turns bullish
            if not alligator_down or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsAlligator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0