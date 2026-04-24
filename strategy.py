#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation.
- Uses Williams %R(14) on 4h timeframe for overbought/oversold reversal signals.
- Long when %R crosses above -80 from below with volume > 1.8x 20-bar average.
- Short when %R crosses below -20 from above with volume > 1.8x 20-bar average.
- Trend filter: price must be above/below 1d EMA34 to align with daily trend.
- Designed for 4h timeframe to capture medium-term swings with controlled trade frequency.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 20-50 trades/year (80-200 total over 4 years) to stay fee-efficient.
- Volume confirmation reduces false reversals in choppy markets.
- Novelty: Williams %R is underutilized in crypto; combines with 1d EMA trend for higher win rate.
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
    
    # Get 4h data ONCE before loop for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R(14) for 4h timeframe
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_4h) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to 15m timeframe (wait for 4h bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms reversal
            if volume_confirm:
                # Long: Williams %R crosses above -80 from below AND price above 1d EMA34
                if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above AND price below 1d EMA34
                elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                      close[i] < ema_34_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR price below 1d EMA34
            if (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR price above 1d EMA34
            if (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0