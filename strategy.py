#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend strength via aligned SMAs
# 1d EMA50 ensures alignment with higher timeframe trend direction
# Volume confirmation requires 1.5x average volume to ensure participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# Works in both bull and bear markets by following the 1d trend direction

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "12h"
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
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator components from 12h data
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using SMA as approximation for SMMA (simple moving average)
    close_series = pd.Series(close)
    jaw = close_series.rolling(window=13, min_periods=13).mean().values
    teeth = close_series.rolling(window=8, min_periods=8).mean().values
    lips = close_series.rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA (balanced)
        vol_series = pd.Series(volume)
        vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Williams Alligator signals with 1d trend filter
        # Long: Lips > Teeth > Jaw (aligned) + volume spike + price above 1d EMA50
        # Short: Lips < Teeth < Jaw (aligned) + volume spike + price below 1d EMA50
        if position == 0:
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and volume_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and volume_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines cross (Lips < Teeth) OR price below 1d EMA50
            if lips[i] < teeth[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines cross (Lips > Teeth) OR price above 1d EMA50
            if lips[i] > teeth[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals