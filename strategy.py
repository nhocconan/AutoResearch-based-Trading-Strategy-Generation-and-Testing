#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R extreme with 1d EMA50 trend filter and volume confirmation
- Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA50 AND volume > 1.5x 20-period average
- Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA50 AND volume > 1.5x 20-period average
- Exit when Williams %R crosses back above -50 (for long) or below -50 (for short)
- Uses 1d EMA50 for trend alignment to avoid counter-trend trades and capture major trend
- Williams %R identifies exhaustion points in both bull and bear markets
- Volume confirmation ensures institutional participation and reduces false signals
- Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R(14) on primary timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(15, 51, 21)  # Need 15 for Williams %R, 51 for EMA50 (50+1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_long = williams_r[i] > -50
        exit_short = williams_r[i] < -50
        
        # Trend filter
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: oversold + uptrend + volume confirmation
            if oversold and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: overbought + downtrend + volume confirmation
            elif overbought and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses back above -50 (for long) or below -50 (for short)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50
                if exit_long:
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R crosses below -50
                if exit_short:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0