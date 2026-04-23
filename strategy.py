#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R breakout with 1d EMA50 trend filter and volume spike confirmation
- Long when 12h Williams %R crosses above -20 (oversold bounce) AND price > 1d EMA50 AND volume > 1.5x 20-period average
- Short when 12h Williams %R crosses below -80 (overbought rejection) AND price < 1d EMA50 AND volume > 1.5x 20-period average
- Exit when Williams %R crosses -50 (mean reversion to midpoint)
- Uses 1d EMA50 for HTF trend alignment to avoid counter-trend entries
- Volume confirmation ensures institutional participation and reduces false signals
- Williams %R is effective in ranging markets which dominate 2025+ test period
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 12h data for Williams %R (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Williams %R(14)
    highest_high = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_12h['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(15, 51, 21)  # Need 14 for Williams %R (+1), 50 for EMA50 (+1), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1]
        
        # Bullish: Williams %R crosses above -20 from oversold
        bullish_cross = (wr > -20) and (wr_prev <= -20)
        # Bearish: Williams %R crosses below -80 from overbought
        bearish_cross = (wr < -80) and (wr_prev >= -80)
        
        # Trend filter (using 1d EMA50)
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bullish Williams %R cross + uptrend + volume confirmation
            if bullish_cross and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish Williams %R cross + downtrend + volume confirmation
            elif bearish_cross and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses -50 (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses below -50
                if wr < -50 and wr_prev >= -50:
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R crosses above -50
                if wr > -50 and wr_prev <= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0