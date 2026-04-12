#!/usr/bin/env python3
"""
6h_1d_Price_Action_Reversal_v1
Hypothesis: On 6h timeframe, price reversals at key 1d support/resistance levels (prior day high/low) 
with volume confirmation and RSI filter provide edge in both bull and bear markets. 
Uses 1d for structure (key levels) and 6s for timely entries. Target: 20-40 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Price_Action_Reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for key levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # === KEY LEVELS: Prior day high and low ===
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    
    # Align to 6h timeframe
    prev_high_6h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_6h = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # === RSI FILTER (14-period on 1d) ===
    delta = pd.Series(daily_close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_6h = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === VOLUME SPIKE (1.5x 20-period average on 6h) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(prev_high_6h[i]) or np.isnan(prev_low_6h[i]) or
            np.isnan(rsi_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price near prior day high/low (within 0.2% tolerance)
        near_high = abs(high[i] - prev_high_6h[i]) / prev_high_6h[i] < 0.002
        near_low = abs(low[i] - prev_low_6h[i]) / prev_low_6h[i] < 0.002
        
        # RSI conditions for reversal
        rsi_oversold = rsi_6h[i] < 30
        rsi_overbought = rsi_6h[i] > 70
        
        # Entry conditions with volume confirmation
        long_entry = near_low and rsi_oversold and vol_spike[i]
        short_entry = near_high and rsi_overbought and vol_spike[i]
        
        # Exit conditions: price moves back toward mid-point or opposite signal
        mid_point = (prev_high_6h[i] + prev_low_6h[i]) / 2.0
        long_exit = close[i] >= mid_point  # Exit long when price reaches midpoint
        short_exit = close[i] <= mid_point  # Exit short when price reaches midpoint
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals