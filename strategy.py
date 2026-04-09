#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with weekly trend filter and volume confirmation
# - Uses 1w Williams %R(14) for extreme oversold/overbought conditions
# - Uses 1d EMA(50) for weekly trend direction (proxy via daily)
# - Enters mean reversion trades on 6h when price touches 6h EMA(20) from extreme %R
# - Volume confirmation: 6h volume > 1.5x 20-period average
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to avoid fee drag
# - Williams %R is effective in ranging markets which appear in both bull and bear regimes

name = "6h_1w_1d_williamsr_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 6h volume average for confirmation
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # 1w Williams %R(14) for extreme conditions
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1w) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1w Williams %R to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # 1d EMA(50) for trend direction (proxy for weekly trend)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 6h EMA(20) for mean reversion target
    close = prices['close'].values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(vol_ma[i]) or
            vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = prices['volume'].iloc[i] > (1.5 * vol_ma[i])
        
        if position == 1:  # Long position
            # Exit when price returns to EMA(20) or Williams %R exits extreme
            if close[i] >= ema_20[i] or williams_r_aligned[i] > -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price returns to EMA(20) or Williams %R exits extreme
            if close[i] <= ema_20[i] or williams_r_aligned[i] < -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries from extreme Williams %R
            # Long: oversold (< -80) in weekly uptrend (price > EMA50)
            if (williams_r_aligned[i] <= -80 and 
                close[i] > ema_50_aligned[i] and 
                volume_confirm):
                position = 1
                signals[i] = 0.25
            # Short: overbought (>= -20) in weekly downtrend (price < EMA50)
            elif (williams_r_aligned[i] >= -20 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals