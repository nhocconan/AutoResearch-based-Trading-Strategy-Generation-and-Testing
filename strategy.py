#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Uses Williams Alligator (jaw/teeth/lips) to identify trending vs ranging markets
# Only trade in Alligator "awake" state (jaws, teeth, lips separated and aligned)
# 1d EMA34 ensures alignment with higher timeframe trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) for 6h timeframe
# Works in bull markets via long signals in uptrend and bear markets via short signals in downtrend

name = "6h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams Alligator on 6h data (using SMAs as per original)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(low).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator calculations)
    start_idx = 20  # buffer for 20-period calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Alligator "awake" state: jaws, teeth, lips separated and aligned
        # For uptrend: lips > teeth > jaw
        # For downtrend: lips < teeth < jaw
        alligator_awake_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_awake_down = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price close > lips + Alligator awake up + 1d close > EMA34 + volume spike
            if (close[i] > lips[i] and alligator_awake_up and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price close < lips + Alligator awake down + 1d close < EMA34 + volume spike
            elif (close[i] < lips[i] and alligator_awake_down and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below lips or Alligator starts sleeping (teeth crosses jaw)
            if close[i] < lips[i] or (teeth[i] <= jaw[i] and lips[i] <= teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above lips or Alligator starts sleeping (teeth crosses jaw)
            if close[i] > lips[i] or (teeth[i] >= jaw[i] and lips[i] >= teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals