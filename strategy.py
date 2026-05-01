#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 or > -20) 
# followed by reversal provide high-probability entries. 12h EMA50 ensures alignment with 
# intermediate trend to avoid counter-trend trades. Volume confirmation filters false signals.
# Designed for 6h timeframe to capture swings with lower frequency (target: 12-30 trades/year).
# Works in both bull and bear markets by following the 12h trend direction.

name = "6h_WilliamsR_Extreme_Reversal_12hEMA50_Trend_VolumeConfirm_v1"
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
    
    # 12h HTF data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period) on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need sufficient history for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Williams %R extremes: oversold < -80, overbought > -20
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Williams %R reversal: crossing back above -80 (long) or below -20 (short)
        # Need previous bar value to detect crossover
        if i > 0:
            prev_williams_r = williams_r[i-1]
            oversold_reversal = (prev_williams_r <= -80) and (williams_r[i] > -80)
            overbought_reversal = (prev_williams_r >= -20) and (williams_r[i] < -20)
        else:
            oversold_reversal = False
            overbought_reversal = False
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R reversal from oversold, volume spike, uptrend
            if oversold_reversal and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R reversal from overbought, volume spike, downtrend
            elif overbought_reversal and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R overbought (> -20) or trend reversal
            if williams_r[i] > -20 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R oversold (< -80) or trend reversal
            if williams_r[i] < -80 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals