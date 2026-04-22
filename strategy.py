#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions with mean-reversion tendency.
# 1d EMA50 trend filter ensures trades align with higher timeframe direction.
# Volume spike (>1.8x 24-period average) confirms momentum behind the reversal.
# Designed for 6h timeframe targeting 15-30 trades/year, effective in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) calculation
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Volume confirmation: 24-period average (4 days of 6h bars)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above 1d EMA50 + volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * vol_avg_24[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + price below 1d EMA50 + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_avg_24[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral zone (-50) or trend reversal
            if position == 1:
                # Exit long: Williams %R returns above -50 or trend turns down
                if (williams_r[i] > -50 or 
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Williams %R returns below -50 or trend turns up
                if (williams_r[i] < -50 or 
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0