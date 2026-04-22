#!/usr/bin/env python3

"""
Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation.
Go long when Williams %R crosses above -80 (oversold) in a 1d uptrend with volume spike.
Go short when Williams %R crosses below -20 (overbought) in a 1d downtrend with volume spike.
Exit when Williams %R returns to the mean (-50) or trend changes.
Williams %R identifies exhaustion points; 1d trend filter avoids counter-trend trades;
volume confirmation ensures participation. Designed for 12-37 trades/year by requiring
oversold/overbought conditions + trend alignment + volume spike.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 12h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend direction
    ema1d_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema1d_34)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema1d_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) + 1d uptrend + volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                ema1d_34_aligned[i] > ema1d_34_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) + 1d downtrend + volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  ema1d_34_aligned[i] < ema1d_34_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to mean (-50) or trend changes
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R >= -50 or 1d trend turns down
                if williams_r[i] >= -50 or ema1d_34_aligned[i] < ema1d_34_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R <= -50 or 1d trend turns up
                if williams_r[i] <= -50 or ema1d_34_aligned[i] > ema1d_34_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_MeanReversion_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0