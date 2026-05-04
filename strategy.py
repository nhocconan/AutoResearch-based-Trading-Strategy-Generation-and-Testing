#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels combined with
# 12h trend alignment and volume spikes provide high-probability entries in both bull and bear markets
# Uses 6h timeframe to balance trade frequency and signal quality, targeting 12-30 trades/year

name = "6h_WilliamsR_Reversal_12hEMA50_VolumeSpike"
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
    
    # Get 12h data for Williams %R calculation and EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R (14-period): (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align to 6h timeframe (wait for completed 12h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 12h EMA50 trend filter from prior completed 12h bar
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_shifted = np.roll(ema50_12h, 1)
    ema50_12h_shifted[0] = np.nan
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 FROM BELOW (oversold reversal)
            # AND 12h EMA50 uptrend AND volume spike
            williams_r_now = williams_r_aligned[i]
            williams_r_prev = williams_r_aligned[i-1]
            ema_trend = close[i] > ema50_12h_aligned[i]
            volume_spike = volume[i] > (2.0 * vol_ema_20[i])
            
            if (williams_r_prev <= -80 and williams_r_now > -80 and 
                ema_trend and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 FROM ABOVE (overbought reversal)
            # AND 12h EMA50 downtrend AND volume spike
            elif (williams_r_prev >= -20 and williams_r_now < -20 and 
                  not ema_trend and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR price below EMA50
            if williams_r_aligned[i] > -20 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR price above EMA50
            if williams_r_aligned[i] < -80 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals