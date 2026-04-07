#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R with 1-week trend filter and volume confirmation
# Hypothesis: Williams %R identifies overbought/oversold conditions, filtered by weekly trend to avoid counter-trend trades.
# Volume confirms institutional participation. Works in bull via pullbacks in uptrend, in bear via bounces in downtrend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
name = "6h_williamsr_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Get daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(wr[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR weekly trend turns down
            if wr[i] > -20 or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR weekly trend turns up
            if wr[i] < -80 or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Williams %R < -80 (oversold) + weekly uptrend + volume confirmation
            if wr[i] < -80 and close[i] > ema_50_1w_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: Williams %R > -20 (overbought) + weekly downtrend + volume confirmation
            elif wr[i] > -20 and close[i] < ema_50_1w_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals