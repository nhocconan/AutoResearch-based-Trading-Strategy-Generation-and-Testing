#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for trend filter (longer-term trend) and 12h Williams %R for momentum extremes
# Entry logic: Long when 12h Williams %R crosses above -80 (oversold bounce) with volume spike and price > 1d EMA34 (uptrend)
#              Short when 12h Williams %R crosses below -20 (overbought rejection) with volume spike and price < 1d EMA34 (downtrend)
# Exit logic: Exit when Williams %R crosses the midpoint (-50) indicating momentum shift, or opposite extreme
# Works in both bull and bear markets by trading mean reversion within the 1d trend
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "12h_WilliamsR_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Williams %R (14-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe (use previous completed 12h bar's value)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA34 (uptrend) AND volume spike
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA34 (downtrend) AND volume spike
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (momentum weakening) OR crosses below -80 (re-oversold)
            if (williams_r_aligned[i] > -50 and williams_r_aligned[i-1] <= -50) or \
               (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (momentum weakening) OR crosses above -20 (re-overbought)
            if (williams_r_aligned[i] < -50 and williams_r_aligned[i-1] >= -50) or \
               (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals