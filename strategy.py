#!/usr/bin/env python3
"""
4h_Chaikin_Money_Flow_Crossover_12hTrend_v1
Hypothesis: On 4h timeframe, use Chaikin Money Flow (CMF) to detect institutional accumulation/distribution.
Enter long when CMF crosses above +0.15 with 12h EMA50 uptrend; short when CMF crosses below -0.15 with 12h EMA50 downtrend.
CMF measures money flow volume, providing early signals of trend changes. Combined with 12h trend filter and volume confirmation,
this reduces whipsaws in choppy markets and captures sustained moves in both bull and bear regimes.
"""
name = "4h_Chaikin_Money_Flow_Crossover_12hTrend_v1"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Chaikin Money Flow (CMF) with period=20
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mfm = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume > 1.5 * 20-period average volume (less strict to avoid over-filtering)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data is not ready
        if (np.isnan(cmf[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            # Minimum 6 bars between trades to reduce frequency (4h timeframe)
            if bars_since_entry < 6:
                continue
                
            # Long: CMF crosses above +0.15 + 12h EMA50 uptrend + volume filter
            if (cmf[i] > 0.15 and cmf[i-1] <= 0.15 and 
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: CMF crosses below -0.15 + 12h EMA50 downtrend + volume filter
            elif (cmf[i] < -0.15 and cmf[i-1] >= -0.15 and 
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Exit: CMF crosses back through zero (loss of momentum)
            if position == 1:
                if cmf[i] < 0 and cmf[i-1] >= 0:  # CMF crossed below zero
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if cmf[i] > 0 and cmf[i-1] <= 0:  # CMF crossed above zero
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals