# 12h Williams Alligator + Volume Spike + Price Channel Breakout
# Combines Williams Alligator (Jaw/Teeth/Lips) for trend direction,
# volume confirmation for breakout strength, and price channel breakouts
# to capture strong momentum moves in both bull and bear markets.
# Uses 1d Williams Alligator to avoid whipsaws and align with higher timeframe trend.
# Target: 15-40 trades/year (60-160 total over 4 years) to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # Need at least 13 days for Alligator
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d data
    # Jaw (blue line): 13-period SMA, shifted 8 bars forward
    # Teeth (red line): 8-period SMA, shifted 5 bars forward  
    # Lips (green line): 5-period SMA, shifted 3 bars forward
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Median price for Alligator calculation
    median_price_1d = (high_1d + low_1d) / 2
    
    # Calculate SMAs
    jaw_raw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    
    # Apply forward shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    jaw[8:] = jaw_raw[:-8]  # Shift forward by 8
    teeth[5:] = teeth_raw[:-5]  # Shift forward by 5
    lips[3:] = lips_raw[:-3]  # Shift forward by 3
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_ltf_to_htf(prices, df_1d, jaw)
    teeth_aligned = align_ltf_to_htf(prices, df_1d, teeth)
    lips_aligned = align_ltf_to_htf(prices, df_1d, lips)
    
    # Calculate 12h price channel (Donchian-like for breakouts)
    lookback = 10  # 10-period lookback for price channel
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Volume spike detection (20-period average)
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period: need enough data for all indicators
    start_idx = max(lookback, vol_period, 13 + 8)  # 13 for jaw, 8 for shift
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Williams Alligator trend detection
        # Bullish alignment: Lips > Teeth > Jaw (green > red > blue)
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish alignment: Jaw > Teeth > Lips (blue > red > green)
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Volume confirmation: spike > 1.8x average (moderate threshold to avoid excessive trades)
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long entry: price breaks above channel in bullish Alligator alignment with volume
            if bullish_alignment and price > highest_high[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below channel in bearish Alligator alignment with volume
            elif bearish_alignment and price < lowest_low[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below channel OR Alligator turns bearish
            if price < lowest_low[i] or bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above channel OR Alligator turns bullish
            if price > highest_high[i] or bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_Volume_Breakout"
timeframe = "12h"
leverage = 1.0