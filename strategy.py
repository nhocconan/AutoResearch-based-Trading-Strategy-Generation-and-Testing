#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Williams Alligator (13,8,5 SMAs with 8,5,3 offsets) identifies trends via jaw-teeth-lips alignment.
# Weekly trend filter prevents counter-trend trades; volume confirms breakout strength.
# Works in bull/bear by only taking trades aligned with weekly trend.
# Target: 30-100 total trades over 4 years (~7-25/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_price_1d = (high_1d + low_1d) / 2  # Williams Alligator uses median price
    
    # Jaw (13-period SMA, 8 bars offset)
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMA, 5 bars offset)
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMA, 3 bars offset)
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 1d timeframe (wait for 1d close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Weekly trend filter: EMA 34 on weekly close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (13 bars for lips), weekly EMA (34), volume MA (20)
    start_idx = max(13, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Alligator alignment: bullish when lips > teeth > jaw
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        # Bearish alignment: lips < teeth < jaw
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Weekly trend filter
        bullish_weekly = price > ema_34_aligned[i]
        bearish_weekly = price < ema_34_aligned[i]
        
        if position == 0:
            # Long: bullish alignment + volume + weekly uptrend
            if bullish_alignment and vol_filter and bullish_weekly:
                signals[i] = size
                position = 1
            # Short: bearish alignment + volume + weekly downtrend
            elif bearish_alignment and vol_filter and bearish_weekly:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish alignment or weekly trend turns down
            if bearish_alignment or not bullish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish alignment or weekly trend turns up
            if bullish_alignment or not bearish_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Williams_Alligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0