#!/usr/bin/env python3
"""
4h Williams Alligator + Volume Spike + RSI Filter
Based on Bill Williams' Alligator indicator (Jaw/Teeth/Lips) to detect trends.
Only takes trades when all three lines are aligned in same direction (trending market).
Adds volume confirmation to avoid false breakouts and RSI filter to avoid overbought/oversold extremes.
Designed for low trade frequency with strong edge in both bull and bear markets by
focusing on strong trending moves only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator (using SMAs as per Williams)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Williams Alligator lines (all SMAs)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw_4h = pd.Series(close_4h).rolling(window=13, min_periods=13).mean()
    jaw_4h = jaw_4h.shift(8)  # shift forward 8 bars
    
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth_4h = pd.Series(close_4h).rolling(window=8, min_periods=8).mean()
    teeth_4h = teeth_4h.shift(5)  # shift forward 5 bars
    
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips_4h = pd.Series(close_4h).rolling(window=5, min_periods=5).mean()
    lips_4h = lips_4h.shift(3)  # shift forward 3 bars
    
    # Align Alligator lines to 4h timeframe (no additional delay needed as Williams uses SMAs)
    jaw_4h_aligned = align_htf_to_ltf(prices, df_4h, jaw_4h.values)
    teeth_4h_aligned = align_htf_to_ltf(prices, df_4h, teeth_4h.values)
    lips_4h_aligned = align_htf_to_ltf(prices, df_4h, lips_4h.values)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # RSI filter (14-period)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_4h_aligned[i]) or np.isnan(teeth_4h_aligned[i]) or 
            np.isnan(lips_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw = jaw_4h_aligned[i]
        teeth = teeth_4h_aligned[i]
        lips = lips_4h_aligned[i]
        
        # Alligator alignment: all three lines in same order
        # Bullish alignment: Lips > Teeth > Jaw (green)
        bullish_align = lips > teeth and teeth > jaw
        # Bearish alignment: Lips < Teeth < Jaw (red)
        bearish_align = lips < teeth and teeth < jaw
        
        if position == 0:
            # Long: Bullish alignment + volume spike + RSI not overbought
            if (bullish_align and 
                volume_spike[i] and 
                rsi[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + volume spike + RSI not oversold
            elif (bearish_align and 
                  volume_spike[i] and 
                  rsi[i] > 30):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: maintain until alignment breaks
            signals[i] = 0.25
            # Exit: Bullish alignment breaks
            if not bullish_align:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: maintain until alignment breaks
            signals[i] = -0.25
            # Exit: Bearish alignment breaks
            if not bearish_align:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Williams_Alligator_Volume_RSI"
timeframe = "4h"
leverage = 1.0