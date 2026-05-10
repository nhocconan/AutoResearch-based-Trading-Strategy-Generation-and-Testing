#2025-07-17: 12h_Williams_Alligator_T147642
# Hypothesis: Williams Alligator on 1d with 12h entry timing for trend-following in both bull and bear markets.
# The Alligator (Jaw=13, Teeth=8, Lips=5) acts as a trend filter; we trade in direction of alignment.
# Entry when price crosses above/below Teeth with volume confirmation; exit when Jaw-Teeth-Lips tangle.
# Uses weekly timeframe for regime filter (only trade when weekly close > weekly EMA50 for longs, < for shorts).
# Designed to avoid whipsaws in ranging markets and capture sustained trends with low trade frequency.

name = "12h_Williams_Alligator_T147642"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Williams Alligator on 1d timeframe: Jaw (13,8), Teeth (8,5), Lips (5,3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Jaw: 13-period SMMA (smoothed moving average) = EMA with alpha=1/13
    jaw = pd.Series(close_1d).ewm(alpha=1/13, adjust=False).mean().values
    # Teeth: 8-period SMMA
    teeth = pd.Series(close_1d).ewm(alpha=1/8, adjust=False).mean().values
    # Lips: 5-period SMMA
    lips = pd.Series(close_1d).ewm(alpha=1/5, adjust=False).mean().values
    
    # Align Alligator lines to 12h timeframe (wait for 1d bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Weekly regime filter: only trade in direction of weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly regime to 12h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Volume confirmation: 20-period average on 12h
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator conditions:
        # Bullish alignment: Lips > Teeth > Jaw (alligator mouth open upward)
        # Bearish alignment: Lips < Teeth < Jaw (alligator mouth open downward)
        # Transitional/tangling: otherwise (avoid trading)
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Price relative to Teeth (entry trigger)
        price_above_teeth = close[i] > teeth_aligned[i]
        price_below_teeth = close[i] < teeth_aligned[i]
        
        # Volume confirmation
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: bullish alignment + price above Teeth + weekly uptrend + volume
            if bullish_alignment and price_above_teeth and weekly_uptrend_aligned[i] > 0.5 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + price below Teeth + weekly downtrend + volume
            elif bearish_alignment and price_below_teeth and weekly_downtrend_aligned[i] > 0.5 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: any disruption of bullish alignment or weekly uptrend
            if not bullish_alignment or weekly_uptrend_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: any disruption of bearish alignment or weekly downtrend
            if not bearish_alignment or weekly_downtrend_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals