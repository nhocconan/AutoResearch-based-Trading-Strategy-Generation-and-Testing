#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R with 200 EMA trend filter and volume confirmation
# Long when Williams %R crosses above -20 (from oversold) with price > EMA200 and volume > 1.5x average
# Short when Williams %R crosses below -80 (from overbought) with price < EMA200 and volume > 1.5x average
# Williams %R identifies momentum reversals, EMA200 filters trend direction, volume confirms strength
# Designed for low frequency (target: 20-40 trades/year) to minimize fee drag in both bull and bear markets

name = "4h_1dWilliamsR_EMA200_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA200 on 4h close
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 1-day Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Williams %R signals: -20 (overbought), -80 (oversold)
    williams_r_signal = williams_r  # we'll use crossover logic
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_signal)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema200[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Williams %R crosses above -20 (exiting oversold) with uptrend and volume
            if (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 and 
                close[i] > ema200[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -80 (exiting overbought) with downtrend and volume
            elif (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 and 
                  close[i] < ema200[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -80 (enter overbought) or trend fails
            if williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -20 (enter oversold) or trend fails
            if williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals