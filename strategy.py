#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA50 trend alignment
- Williams Alligator: Jaw (SMA13 shifted 8), Teeth (SMA8 shifted 5), Lips (SMA5 shifted 3)
- Long when Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA50 + volume confirmation
- Short when Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA50 + volume confirmation
- Volume confirmation: current volume > 1.5 * 20-period volume MA
- Discrete signal size: 0.25
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Alligator catches trending moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Jaw: SMA(13) shifted 8 bars
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8) shifted 5 bars
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5) shifted 3 bars
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe (primary)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    # Alligator alignment: bullish when Lips > Teeth > Jaw, bearish when Lips < Teeth < Jaw
    bullish_align = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    bearish_align = (lips_aligned < teeth_aligned) & (teeth_aligned < jaw_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # Need 1d EMA50, volume MA(20), and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish Alligator alignment AND uptrend AND volume confirmation
            if bullish_align[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND downtrend AND volume confirmation
            elif bearish_align[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment turns bearish or reverse signal
            if not bullish_align[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment turns bullish or reverse signal
            if not bearish_align[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0