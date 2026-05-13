#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) crossover with 1d EMA50 trend filter and volume confirmation (>1.3x 20-bar avg volume).
# Uses Alligator for trend identification and momentum, EMA50 for higher timeframe alignment, volume spike for participation.
# Designed for BTC/ETH with discrete sizing (0.25) to minimize fee churn while capturing sustained trends.
# Target: 50-150 total trades over 4 years on 12h timeframe.

name = "12h_Williams_Alligator_1dEMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (using close prices)
    jaw_period, jaw_shift = 13, 8   # Blue line
    teeth_period, teeth_shift = 8, 5  # Red line
    lips_period, lips_shift = 5, 3    # Green line
    
    close_series = pd.Series(close)
    jaw = close_series.ewm(span=jaw_period, adjust=False).mean().shift(jaw_shift).values
    teeth = close_series.ewm(span=teeth_period, adjust=False).mean().shift(teeth_shift).values
    lips = close_series.ewm(span=lips_period, adjust=False).mean().shift(lips_shift).values
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(jaw_shift, teeth_shift, lips_shift, lookback_vol), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment), close > 1d EMA50, volume spike
            if (lips[i] > teeth[i] and 
                teeth[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25  # Enter long
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment), close < 1d EMA50, volume spike
            elif (lips[i] < teeth[i] and 
                  teeth[i] < jaw[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.3 * avg_volume[i]):
                signals[i] = -0.25  # Enter short
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Lips cross below Teeth or volume drops
            if lips[i] < teeth[i] or volume[i] < avg_volume[i]:
                signals[i] = 0.0  # Exit
                position = 0
            else:
                signals[i] = 0.25  # Hold long
        elif position == -1:
            # EXIT SHORT: Lips cross above Teeth or volume drops
            if lips[i] > teeth[i] or volume[i] < avg_volume[i]:
                signals[i] = 0.0  # Exit
                position = 0
            else:
                signals[i] = -0.25  # Hold short
    
    return signals