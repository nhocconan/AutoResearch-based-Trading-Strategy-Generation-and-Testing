#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above R1 in 1d uptrend with volume > 2.0x 20-period MA.
Short when price breaks below S1 in 1d downtrend with volume > 2.0x 20-period MA.
Exit when price returns to the 1d EMA34 level.
Uses 1d HTF for trend alignment with 12h bars. Designed for ~12-30 trades/year with strong edge in both bull and bear markets via trend filter and Camarilla's mean-reversion nature.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d high, low, close for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d close > EMA34 = uptrend, close < EMA34 = downtrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 12h volume > 2.0x 20-period MA (strong confirmation)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakdown_s1 = close[i] < s1_aligned[i]
        return_to_ema = abs(close[i] - ema_34_1d_aligned[i]) < (ema_34_1d_aligned[i] * 0.001)  # within 0.1% of EMA
        
        if position == 0:
            # Long: Price breaks above R1 AND uptrend AND volume spike
            if breakout_r1 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 AND downtrend AND volume spike
            elif breakdown_s1 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Price returns to EMA34 level
            exit_signal = False
            
            if position == 1:
                # Long exit: Price returns to EMA34
                if return_to_ema:
                    exit_signal = True
            elif position == -1:
                # Short exit: Price returns to EMA34
                if return_to_ema:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0