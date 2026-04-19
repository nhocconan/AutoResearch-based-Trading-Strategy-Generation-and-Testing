#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA13 trend filter and volume confirmation
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# When all three are aligned (Jaw > Teeth > Lips for uptrend, reverse for downtrend), trend is strong.
# Uses 1d EMA13 for higher timeframe trend bias and volume confirmation to filter weak signals.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "12h_WilliamsAlligator_1dEMA13_Volume"
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
    
    # 1d EMA13 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Williams Alligator on 12h: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_13_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Jaw > Teeth > Lips (bullish alignment) + above 1d EMA13 + volume confirmation
            if (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                close[i] > ema_13_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + below 1d EMA13 + volume confirmation
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_13_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator alignment breaks (teeth <= lips) or breaks below 1d EMA13
            if (teeth[i] <= lips[i]) or (close[i] < ema_13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator alignment breaks (jaw <= teeth) or breaks above 1d EMA13
            if (jaw[i] <= teeth[i]) or (close[i] > ema_13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals