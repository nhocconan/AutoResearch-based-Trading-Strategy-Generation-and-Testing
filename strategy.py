#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + EMA50 trend filter + volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend absence (all lines intertwined) vs presence (lines separated, ordered)
# In trending markets: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
# In ranging markets: Alligator lines are tangled/intertwined
# Combined with 1w EMA50 for higher timeframe trend alignment and volume spike confirmation
# Discrete sizing 0.25 to limit fee drag; target 30-100 trades over 4 years
# Williams Alligator is effective in both bull and bear markets by filtering choppy conditions

name = "1d_WilliamsAlligator_1wEMA50_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 13 or len(df_1w) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    # SMMA = Smoothed Moving Average (similar to EMA but with different smoothing)
    close_1d_series = pd.Series(close_1d)
    jaw = close_1d_series.ewm(alpha=1/13, adjust=False).mean().shift(8).values
    teeth = close_1d_series.ewm(alpha=1/8, adjust=False).mean().shift(5).values
    lips = close_1d_series.ewm(alpha=1/5, adjust=False).mean().shift(3).values
    
    # Calculate 1w EMA50 trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume spike confirmation (volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * volume_ma)
    
    # Align HTF indicators to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator condition: lines separated and ordered (not intertwined)
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Lips < Teeth < Jaw
        alligator_long = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long entry: Alligator uptrend + price above EMA50 + volume spike
            if alligator_long and close[i] > ema50_1w_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator downtrend + price below EMA50 + volume spike
            elif alligator_short and close[i] < ema50_1w_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator trend reversal (lines start to intertwine or reverse order)
            if not alligator_long:  # Lips <= Teeth or Teeth <= Jaw
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator trend reversal (lines start to intertwine or reverse order)
            if not alligator_short:  # Lips >= Teeth or Teeth >= Jaw
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals