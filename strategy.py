#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# When the three lines are intertwined (no clear trend), we stay out (choppy market).
# When they diverge in alignment (Lips > Teeth > Jaw for up, reverse for down), we trade.
# 1d EMA50 ensures we trade only in the direction of the daily trend.
# Volume confirmation ensures we only enter on strong moves.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
# Works in bull markets (Alligator aligned up with uptrend) and bear markets (aligned down with downtrend).
name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = close_s.rolling(window=8, min_periods=8).mean().values   # Red line
    lips = close_s.rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_down = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Alligator aligned up AND uptrend AND volume confirmation
            if alligator_up and uptrend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down AND downtrend AND volume confirmation
            elif alligator_down and downtrend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks OR trend reverses
            if not alligator_up or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks OR trend reverses
            if not alligator_down or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals