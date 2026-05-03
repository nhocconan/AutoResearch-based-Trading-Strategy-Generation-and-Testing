#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) trend filter with volume confirmation
# Long when price > Alligator Lips and Jaw > Teeth > Lips (bullish alignment) and volume > 1.5x 20-bar average
# Short when price < Alligator Lips and Jaw < Teeth < Lips (bearish alignment) and volume > 1.5x 20-bar average
# Exit via ATR(14) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses Williams Alligator from daily timeframe for trend structure, volume for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 50-150 total trades over 4 years = 12-37/year.

name = "12h_WilliamsAlligator_1d_Volume_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Williams Alligator (using SMAs with specific offsets)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # Need at least 13 bars for Alligator calculation
        return np.zeros(n)
    
    # Calculate SMAs for Alligator
    # Jaw: 13-period SMA shifted 8 bars forward
    jaw = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted 5 bars forward
    teeth = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted 3 bars forward
    lips = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components to 12h timeframe (completed 1d bar only)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for Alligator and ATR calculations)
    start_idx = 50  # Alligator needs ~13+8=21 bars, plus buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Lips and bullish alignment (Jaw > Teeth > Lips) and volume spike
            if (close[i] > lips_aligned[i] and 
                jaw_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price < Lips and bearish alignment (Jaw < Teeth < Lips) and volume spike
            elif (close[i] < lips_aligned[i] and 
                  jaw_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < lips_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.5 * ATR
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals