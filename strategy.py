#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 12h EMA50 trend filter and volume confirmation.
Williams %R identifies overbought/oversold conditions for mean reversion entries.
12h EMA50 provides trend filter to avoid counter-trend trades in strong trends.
Volume confirmation ensures breakout/mean reversion has participation.
Designed for 4h timeframe to balance trade frequency (target: 20-50 trades/year).
Uses discrete position sizing (0.25) to manage fee drag and drawdown.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R (14 period) on 4h timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_4h) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Calculate volume spike: current volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # need EMA50, Williams %R, and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND uptrend on 12h AND volume spike
            if williams_r_aligned[i] < -80 and trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend on 12h AND volume spike
            elif williams_r_aligned[i] > -20 and trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50 to -50) or opposite extreme
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R rises above -50 (mean reversion complete)
                if williams_r_aligned[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R falls below -50 (mean reversion complete)
                if williams_r_aligned[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_MeanReversion_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0