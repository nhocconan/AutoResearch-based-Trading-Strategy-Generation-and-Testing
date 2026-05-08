#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1-day trend filter and volume confirmation.
# Uses Alligator (Jaw/Teeth/Lips) for trend identification in 12h timeframe.
# Long when Lips > Teeth > Jaw (bullish alignment) + price > 1-day EMA50 + volume spike.
# Short when Lips < Teeth < Jaw (bearish alignment) + price < 1-day EMA50 + volume spike.
# Designed for fewer trades (target: 15-30/year) to minimize fee drag on 12h timeframe.
# Works in both bull and bear markets by requiring trend alignment and volume confirmation.

name = "12h_Williams_Alligator_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Williams Alligator (13,8,5 SMAs shifted)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Jaw: 13-period SMA, shifted 8 bars
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    jaw_12h = np.roll(jaw_12h, 8)
    jaw_12h[:8] = np.nan
    # Teeth: 8-period SMA, shifted 5 bars
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    teeth_12h = np.roll(teeth_12h, 5)
    teeth_12h[:5] = np.nan
    # Lips: 5-period SMA, shifted 3 bars
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    lips_12h = np.roll(lips_12h, 3)
    lips_12h[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # 1-day trend: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator: Lips > Teeth > Jaw
            bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
            # Bearish Alligator: Lips < Teeth < Jaw
            bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
            
            # Long: bullish alignment + price > 1d EMA50 + volume spike
            long_cond = bullish_alignment and (close[i] > ema_50_1d_aligned[i]) and volume_spike[i]
            # Short: bearish alignment + price < 1d EMA50 + volume spike
            short_cond = bearish_alignment and (close[i] < ema_50_1d_aligned[i]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish (Lips < Teeth)
            if lips_aligned[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish (Lips > Teeth)
            if lips_aligned[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals