#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Williams %R (14) extreme + 1w EMA50/EMA200 trend filter + volume confirmation.
Long when Williams %R < -80 (oversold) with 1w EMA50 > EMA200 (uptrend) and volume > 1.3x 20-period volume average.
Short when Williams %R > -20 (overbought) with 1w EMA50 < EMA200 (downtrend) and volume > 1.3x 20-period volume average.
Exit on opposite Williams %R extreme (%R > -50 for long exit, %R < -50 for short exit).
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years (12-37/year).
Williams %R captures mean reversal in overextended moves; 1w EMA filter ensures trading with primary trend only;
volume confirmation ensures institutional participation. Designed to work in bull markets (buy dips in uptrend)
and bear markets (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Williams %R(14) on 1d data
    def williams_r(high_vals, low_vals, close_vals, window):
        highest_high = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close_vals) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        wr = np.where((highest_high - lowest_low) == 0, -50, wr)
        return wr
    
    wr_14_1d = williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1w EMA50 and EMA200 for trend
    def ema(values, span):
        return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema_50_1w = ema(close_1w, 50)
    ema_200_1w = ema(close_1w, 200)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (12h)
    wr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_14_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R and EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_14_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20_1d_aligned[i]
        # Trend filter: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
        uptrend = ema_50_1w_aligned[i] > ema_200_1w_aligned[i]
        downtrend = ema_50_1w_aligned[i] < ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with uptrend and volume
            if (wr_14_1d_aligned[i] < -80 and 
                uptrend and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with downtrend and volume
            elif (wr_14_1d_aligned[i] > -20 and 
                  downtrend and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -50 (exiting oversold territory)
            if wr_14_1d_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -50 (exiting overbought territory)
            if wr_14_1d_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dWilliamsR14_1wEMA50_200_Volume_Confirm"
timeframe = "12h"
leverage = 1.0