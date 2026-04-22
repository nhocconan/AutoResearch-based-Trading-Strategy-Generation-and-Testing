#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d weekly 52-week high/low breakout with volume confirmation
    # Weekly 52-week high/low represent major institutional support/resistance
    # Breakouts with volume confirm institutional participation
    # Works in bull/bear: breaks key yearly levels with volume confirmation
    # Low trade frequency (< 25/year) minimizes fee drag
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for 52-week high/low calculation
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate 52-week high/low (260 trading days ~ 52 weeks)
    # Using weekly data: 52 weeks lookback
    period = 52
    high_52w = pd.Series(high_weekly).rolling(window=period, min_periods=period).max().values
    low_52w = pd.Series(low_weekly).rolling(window=period, min_periods=period).min().values
    
    # Align 52-week levels to daily timeframe
    high_52w_aligned = align_htf_to_ltf(prices, df_weekly, high_52w)
    low_52w_aligned = align_htf_to_ltf(prices, df_weekly, low_52w)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_52w_aligned[i]) or 
            np.isnan(low_52w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 52-week high with volume spike
            if close[i] > high_52w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below 52-week low with volume spike
            elif close[i] < low_52w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite 52-week level
            if position == 1:
                if close[i] < low_52w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_52w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_52Week_High_Low_Breakout_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0