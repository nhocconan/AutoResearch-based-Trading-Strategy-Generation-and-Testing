#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (3 SMAs) with 1w trend filter and volume confirmation.
# The Alligator uses SMA(13,8,5) to identify trend phases (sleeping, waking, feeding).
# Combined with 1w EMA trend filter to avoid counter-trend trades and volume spikes for confirmation.
# Works in both bull and bear markets by taking long only when price > 1w EMA and Alligator aligned up,
# short only when price < 1w EMA and Alligator aligned down.
# Target: 7-25 trades per year (30-100 total over 4 years) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Daily data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1-week EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema21_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (21 + 1)
    ema21_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema21_1w[i] = (close_1w[i] - ema21_1w[i-1]) * ema_multiplier + ema21_1w[i-1]
    
    # Calculate Williams Alligator components: Jaw (13), Teeth (8), Lips (5) SMAs
    close_1d = df_1d['close'].values
    jaw = np.zeros(len(close_1d))
    teeth = np.zeros(len(close_1d))
    lips = np.zeros(len(close_1d))
    
    # Jaw: SMA(13)
    for i in range(len(close_1d)):
        if i < 12:
            jaw[i] = np.nan
        else:
            jaw[i] = np.mean(close_1d[i-12:i+1])
    
    # Teeth: SMA(8)
    for i in range(len(close_1d)):
        if i < 7:
            teeth[i] = np.nan
        else:
            teeth[i] = np.mean(close_1d[i-7:i+1])
    
    # Lips: SMA(5)
    for i in range(len(close_1d)):
        if i < 4:
            lips[i] = np.nan
        else:
            lips[i] = np.mean(close_1d[i-4:i+1])
    
    # Align all indicators to 1d timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate average volume (30-period = 30 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(30, n):
        avg_volume[i] = np.mean(volume[i-30:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema21_1w_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        # Alligator alignment: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        bullish_aligned = lips_val > teeth_val and teeth_val > jaw_val
        bearish_aligned = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long: Bullish alignment + above weekly EMA21 + volume confirmation
            if (bullish_aligned and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Bearish alignment + below weekly EMA21 + volume confirmation
            elif (bearish_aligned and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns bearish or trend turns down
            if (not bullish_aligned or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator turns bullish or trend turns up
            if (not bearish_aligned or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_WilliamsAlligator_Trend_Volume"
timeframe = "1d"
leverage = 1.0