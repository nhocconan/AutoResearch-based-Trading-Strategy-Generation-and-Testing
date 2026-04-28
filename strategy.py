#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme levels with volume confirmation and 4h EMA50 trend filter.
# Enter long when Williams %R crosses above -80 (oversold recovery) with volume > 1.8x 20-bar average and close > EMA50 (uptrend).
# Enter short when Williams %R crosses below -20 (overbought rejection) with volume > 1.8x 20-bar average and close < EMA50 (downtrend).
# Exit on opposite Williams %R level (-20 for long, -80 for short) or EMA50 cross.
# Williams %R captures mean reversion in extremes, volume confirms momentum, EMA50 filters trend.
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets.

name = "4h_WilliamsR_1d_Extreme_Volume_EMA50_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14)
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions with volume confirmation and trend filter
        # Long: Williams %R crosses above -80 (recovery from oversold)
        williams_r_cross_up_80 = (williams_r_aligned[i] > -80) and (williams_r_aligned[i-1] <= -80)
        long_condition = williams_r_cross_up_80 and volume_confirm[i] and close[i] > ema_50[i]
        
        # Short: Williams %R crosses below -20 (rejection from overbought)
        williams_r_cross_down_20 = (williams_r_aligned[i] < -20) and (williams_r_aligned[i-1] >= -20)
        short_condition = williams_r_cross_down_20 and volume_confirm[i] and close[i] < ema_50[i]
        
        # Exit conditions: opposite Williams %R level or EMA50 cross
        long_exit = (williams_r_aligned[i] >= -20) or (close[i] < ema_50[i])
        short_exit = (williams_r_aligned[i] <= -80) or (close[i] > ema_50[i])
        
        # Handle entries and exits
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals