#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Williams Alligator (Jaw, Teeth, Lips) identifies trend direction and strength.
# In trending markets (price > Jaw), trade pullbacks to Teeth/Lips with volume.
# Uses weekly trend filter to avoid counter-trend trades in strong trends.
# Target: 30-100 total trades over 4 years (~7-25/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Williams Alligator on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = pd.Series(median_price_1d).rolling(window=13, center=False).mean().shift(8).values
    teeth = pd.Series(median_price_1d).rolling(window=8, center=False).mean().shift(5).values
    lips = pd.Series(median_price_1d).rolling(window=5, center=False).mean().shift(3).values
    
    # Align Alligator lines to 1d timeframe (wait for 1d close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Weekly trend filter: price above/below weekly EMA(8)
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    # Volume filter: volume > 1.5 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator lines (13+8=21), weekly EMA (8), volume MA (20)
    start_idx = max(21, 8, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_8_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter
        bullish_weekly = price > ema_8_aligned[i]
        bearish_weekly = price < ema_8_aligned[i]
        
        # Alligator alignment: all three lines ordered (trending market)
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Check for proper alignment (alligator sleeping vs awake)
        bullish_aligned = lips_val > teeth_val > jaw_val  # Lips > Teeth > Jaw
        bearish_aligned = lips_val < teeth_val < jaw_val  # Lips < Teeth < Jaw
        
        if position == 0:
            # Long: price > Jaw + bullish alignment + volume + weekly bullish
            if (price > jaw_val and bullish_aligned and vol_filter and bullish_weekly):
                signals[i] = size
                position = 1
            # Short: price < Jaw + bearish alignment + volume + weekly bearish
            elif (price < jaw_val and bearish_aligned and vol_filter and bearish_weekly):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Teeth or weekly trend turns bearish
            if price < teeth_aligned[i] or not bullish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Teeth or weekly trend turns bullish
            if price > teeth_aligned[i] or not bearish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Williams_Alligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0