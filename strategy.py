#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 12h trend filter and volume confirmation
# Uses Williams %R(14) to identify oversold (< -80) and overbought (> -20) conditions
# Only takes reversals in the direction of the 12h EMA(50) trend to avoid counter-trend trades
# Volume spike (1.8x 24-period average) confirms participation and reduces false signals
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Williams %R is effective in ranging markets and captures reversals in bear rallies
# 12h trend filter provides strong directional bias suitable for 6h timeframe

name = "6h_WilliamsR_Extreme_12hTrend_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for Williams %R and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h Williams %R(14)
    highest_high = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    close_12h = df_12h['close'].values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 6h volume confirmation (1.8x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R, EMA and volume MA)
    start_idx = 80  # max(24 for volume, 50 for EMA, 14 for Williams) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) AND uptrend AND volume confirm
            if (williams_r_aligned[i] < -80 and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend AND volume confirm
            elif (williams_r_aligned[i] > -20 and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -50 (momentum fading) OR trend reverses to downtrend
            if (williams_r_aligned[i] > -50 or 
                not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (momentum fading) OR trend reverses to uptrend
            if (williams_r_aligned[i] < -50 or 
                not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals