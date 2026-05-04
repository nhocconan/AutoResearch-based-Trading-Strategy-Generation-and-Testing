#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extremes with 1d EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Enters long when %R crosses above -80 from below
# (oversold bounce) with volume confirmation and 1d EMA50 uptrend. Enters short when %R crosses below -20
# from above (overbought reversal) with volume confirmation and 1d EMA50 downtrend.
# Designed for 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
# Williams %R works in ranging markets (mean reversion) and trends (pullbacks). Volume confirms momentum.
# EMA50 filter ensures trades align with higher timeframe trend, reducing false signals in chop.

name = "6h_WilliamsR_Extremes_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA50 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1d data for EMA50 trend filter - ONCE before loop
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (oversold bounce) AND volume spike AND 1d EMA50 uptrend
            if (williams_r_aligned[i] > -80 and 
                williams_r_aligned[i-1] <= -80 and  # crossed above -80
                volume_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 (overbought reversal) AND volume spike AND 1d EMA50 downtrend
            elif (williams_r_aligned[i] < -20 and 
                  williams_r_aligned[i-1] >= -20 and  # crossed below -20
                  volume_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum loss) OR trend reverses
            if williams_r_aligned[i] < -50 and williams_r_aligned[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema50_1d_aligned[i]:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum loss) OR trend reverses
            if williams_r_aligned[i] > -50 and williams_r_aligned[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema50_1d_aligned[i]:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals