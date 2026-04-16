#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and 1w EMA50 trend filter
# Long when price > Alligator Jaw AND Jaw > Teeth AND Teeth > Lips (bullish alignment)
# Short when price < Alligator Jaw AND Jaw < Teeth AND Teeth < Lips (bearish alignment)
# Williams Alligator uses smoothed medians (13,8,5 periods) with future shifts (8,5,3)
# EMA50 filter ensures alignment with long-term trend
# Volume confirmation adds conviction to signals
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w EMA50 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # === 1d Williams Alligator (using median price) ===
    df_1d = get_htf_data(prices, '1d')
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3) - all smoothed with future shift
    jaw = pd.Series(median_1d).ewm(span=13, adjust=False).mean().values
    jaw = np.roll(jaw, 8)  # future shift 8
    teeth = pd.Series(median_1d).ewm(span=8, adjust=False).mean().values
    teeth = np.roll(teeth, 5)  # future shift 5
    lips = pd.Series(median_1d).ewm(span=5, adjust=False).mean().values
    lips = np.roll(lips, 3)  # future shift 3
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 1d Volume Confirmation ===
    vol_ma_1d = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 periods of 1h = 1d (12h data)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_confirm = volume[i] > vol_ma_1d[i] * 1.5  # 1.5x average volume for confirmation
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit if bearish alignment breaks OR price below EMA50
            if not (jaw_val > teeth_val and teeth_val > lips_val) or price < ema_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit if bullish alignment breaks OR price above EMA50
            if not (jaw_val < teeth_val and teeth_val < lips_val) or price > ema_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bullish alignment: Jaw > Teeth > Lips + price above Jaw + above EMA50 + volume
            if jaw_val > teeth_val and teeth_val > lips_val and price > jaw_val and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Bearish alignment: Jaw < Teeth < Lips + price below Jaw + below EMA50 + volume
            elif jaw_val < teeth_val and teeth_val < lips_val and price < jaw_val and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1dVolume1.5x_1wEMA50_TrendFilter"
timeframe = "12h"
leverage = 1.0