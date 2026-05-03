#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA trend filter and volume confirmation
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) from 6h for trend identification
# 1d EMA34 for higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation requires 1.5x average volume to ensure strong participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
# Alligator works in ranging markets (lines intertwined) and trends (lines diverge)
# EMA filter ensures we only trade with the dominant 1d trend
# Prioritizes BTC/ETH performance with SOL as secondary

name = "6h_WilliamsAlligator_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator from 6h data (smoothed medians)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    def smma(source, period):
        """Smoothed Moving Average"""
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + PRICE) / PERIOD
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align Alligator lines to current timeframe (they're already on 6h)
    jaw_aligned = jaw  # No alignment needed as calculated on same TF
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Alligator signals:
        # Bullish: Lips > Teeth > Jaw (lines diverging upward)
        # Bearish: Lips < Teeth < Jaw (lines diverging downward)
        # Range: Lines intertwined (not clearly ordered)
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_aligned = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # 1d trend filter: price above/below EMA34
        uptrend_1d = close[i] > ema_34_1d_aligned[i]
        downtrend_1d = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Enter long: Bullish Alligator + volume spike + 1d uptrend
            if bullish_aligned and volume_spike and uptrend_1d:
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish Alligator + volume spike + 1d downtrend
            elif bearish_aligned and volume_spike and downtrend_1d:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish Alligator formed OR 1d trend turns down
            if bearish_aligned or downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish Alligator formed OR 1d trend turns up
            if bullish_aligned or uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals