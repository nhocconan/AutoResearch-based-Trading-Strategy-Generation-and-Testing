#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA(50) trend filter and volume confirmation
# Williams Alligator consists of three SMAs: Jaw (13,8), Teeth (8,5), Lips (5,3).
# In strong trends, the Alligator lines are ordered and separated (Jaw > Teeth > Lips for uptrend).
# We use 1w EMA(50) for higher timeframe trend filter and require Alligator alignment in same direction.
# Volume confirmation (1.5x 20-period average) ensures institutional participation.
# Designed for very low trade frequency (7-25/year) on 1d timeframe to minimize fee drag.
# Works in both bull and bear markets by following the dominant trend.

name = "1d_WilliamsAlligator_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator components on 1d
    # Jaw: 13-period SMMA smoothed by 8 periods
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    # Teeth: 8-period SMMA smoothed by 5 periods
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    # Lips: 5-period SMMA smoothed by 3 periods
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    # Volume confirmation (1.5x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(50 for 1w EMA, 13+8 for jaw, 8+5 for teeth, 5+3 for lips, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator aligned upward (Lips > Teeth > Jaw) + price above 1w EMA(50) + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator aligned downward (Lips < Teeth < Jaw) + price below 1w EMA(50) + volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator loses upward alignment (Lips <= Teeth or Teeth <= Jaw) or price below 1w EMA(50)
            if (lips[i] <= teeth[i] or teeth[i] <= jaw[i]) or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator loses downward alignment (Lips >= Teeth or Teeth >= Jaw) or price above 1w EMA(50)
            if (lips[i] >= teeth[i] or teeth[i] >= jaw[i]) or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals