#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Reversal with 1d Elder Ray filter and volume confirmation.
Long when Williams %R(14) crosses above -80 (oversold) AND 1d Bear Power < 0 AND volume > 1.5x 20-period average.
Short when Williams %R(14) crosses below -20 (overbought) AND 1d Bull Power > 0 AND volume > 1.5x 20-period average.
Exit when Williams %R crosses back through -50 (mean reversion) or opposing signal appears.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Williams %R identifies exhaustion points in both bull and bear markets. 1d Elder Ray confirms underlying bull/bear power alignment.
Volume confirmation ensures institutional participation. Designed for 6h timeframe to balance signal quality and trade frequency.
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
    
    # Load 1d data for Elder Ray filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray (standard setting)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Calculate Williams %R(14) on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high==low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align HTF Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 14, 20)  # Ensure warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND 1d Bear Power < 0 AND volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and  # crossed above -80
                bear_power_1d_aligned[i] < 0 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND 1d Bull Power > 0 AND volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and  # crossed below -20
                  bull_power_1d_aligned[i] > 0 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R crosses back through -50 (mean reversion)
            if position == 1 and williams_r[i] < -50 and williams_r[i-1] >= -50:  # crossed below -50
                exit_signal = True
            elif position == -1 and williams_r[i] > -50 and williams_r[i-1] <= -50:  # crossed above -50
                exit_signal = True
            
            # Secondary exit: opposing Williams %R signal with volume confirmation
            elif position == 1 and williams_r[i] < -20 and williams_r[i-1] >= -20 and volume[i] > 1.5 * vol_ma_val:
                exit_signal = True  # overbought reversal
            elif position == -1 and williams_r[i] > -80 and williams_r[i-1] <= -80 and volume[i] > 1.5 * vol_ma_val:
                exit_signal = True  # oversold reversal
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dElderRay_VolumeConfirm"
timeframe = "6h"
leverage = 1.0