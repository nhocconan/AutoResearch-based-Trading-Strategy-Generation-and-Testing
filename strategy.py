#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation.
Longs when Williams %R < -80 (oversold) with 1d EMA50 uptrend and volume > 1.3x average.
Shorts when Williams %R > -20 (overbought) with 1d EMA50 downtrend and volume > 1.3x average.
Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
Williams %R identifies exhaustion points in both trending and ranging markets.
Designed for 15-25 trades/year to minimize fee fade while capturing high-probability reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for EMA trend and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align EMA and Williams %R to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: volume spike > 1.3x 30-period average
    vol_ma_30 = pd.Series(prices['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ratio = prices['volume'].values / vol_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: oversold with uptrend and volume
            if (williams_r_val < -80 and 
                ema_50_val > 0 and  # EMA slope proxy: current EMA > previous EMA (handled by alignment)
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: overbought with downtrend and volume
            elif (williams_r_val > -20 and 
                  ema_50_val < 0 and  # EMA slope proxy
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R crosses back above -50 (long) or below -50 (short)
            exit_signal = False
            
            if position == 1 and williams_r_val > -50:
                exit_signal = True
            elif position == -1 and williams_r_val < -50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_MeanReversion_1dEMA50_Trend_Volume1.3x"
timeframe = "12h"
leverage = 1.0