# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator uses three SMAs (Jaw 13, Teeth 8, Lips 5) to detect trends.
# In trending markets: Lips > Teeth > Jaw (bull) or Lips < Teeth < Jaw (bear).
# Adding 1d EMA(50) filter ensures we only trade in higher timeframe trend direction.
# Volume confirmation (>1.5x 20-bar median) ensures institutional participation.
# Designed for 12h timeframe to capture multi-day trends with low trade frequency.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator SMAs on 12h data
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup for Williams Alligator
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Williams Alligator conditions
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # 1d trend filter: price above/below EMA50
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Long: Williams Alligator bullish + 1d uptrend + volume spike
        if (bullish_alignment and uptrend_1d and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Williams Alligator bearish + 1d downtrend + volume spike
        elif (bearish_alignment and downtrend_1d and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Williams Alligator loses alignment (market entering consolidation)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and not bullish_alignment) or
               (signals[i-1] == -0.25 and not bearish_alignment))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0