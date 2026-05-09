#!/usr/bin/env python3
# Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold) with EMA50 uptrend and volume > 1.5x average
# Short when Williams %R crosses below -20 (overbought) with EMA50 downtrend and volume > 1.5x average
# Exit when Williams %R crosses opposite threshold (-20 for long, -80 for short)
# Williams %R identifies overextended moves, EMA50 filters trend direction, volume confirms conviction
# Designed for low-frequency, high-conviction trades on 12h timeframe
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_WilliamsR_EMA50_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and Williams %R calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 (oversold), EMA50 uptrend, volume confirmation
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 (overbought), EMA50 downtrend, volume confirmation
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses below -20 (overbought threshold)
            if williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -80 (oversold threshold)
            if williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals