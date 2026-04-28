#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R regime filter. 
# Uses Bull Power (high - EMA13) and Bear Power (EMA13 - low) with 1d Williams %R extremes.
# Long when Bull Power > 0, Bear Power < 0, and 1d Williams %R < -80 (oversold).
# Short when Bull Power < 0, Bear Power > 0, and 1d Williams %R > 20 (overbought).
# Exit when power signals reverse or Williams %R returns to neutral zone (-50 to 50).
# Works in bull markets via power strength and in bear markets via faded extremes.
# Target: 50-150 total trades over 4 years (12-37/year).
# Uses discrete sizing (0.25) to limit drawdown and fee churn.

name = "6h_ElderRay_WilliamsR_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Williams %R regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)
    
    # Align Williams %R to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 6h EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Ensure sufficient history for EMA13
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        williams_oversold = williams_r_aligned[i] < -80
        williams_overbought = williams_r_aligned[i] > -20
        williams_neutral = (williams_r_aligned[i] >= -50) and (williams_r_aligned[i] <= 50)
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] > 0
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: bull power > 0, bear power < 0, Williams oversold
            if bull_strong and (not bear_strong) and williams_oversold:
                signals[i] = 0.25
                position = 1
            # Short entry: bull power < 0, bear power > 0, Williams overbought
            elif (not bull_strong) and bear_strong and williams_overbought:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit on reversal
            # Exit if bull power turns negative OR bear power turns positive OR Williams returns to neutral
            if (not bull_strong) or bear_strong or williams_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit on reversal
            # Exit if bull power turns positive OR bear power turns negative OR Williams returns to neutral
            if bull_strong or (not bear_strong) or williams_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals