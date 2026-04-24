#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme Reversal with 1w ADX Trend Filter and Volume Spike.
- Primary timeframe: 12h for execution, HTF: 1w for ADX trend filter.
- Entry: Williams %R(14) crosses above -20 from below (long) or below -80 from above (short) with volume > 2.0x 20-period volume MA.
- Direction filter: only long when 1w ADX > 25 and +DI > -DI (strong uptrend), only short when 1w ADX > 25 and -DI > +DI (strong downtrend).
- Williams %R identifies overbought/oversold conditions; ADX filters for trending markets to avoid whipsaws.
- Volume confirmation reduces false signals.
- Exit: Williams %R returns to -50 (mean reversion) or ADX < 20 (trend weakening).
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying pullbacks in uptrend, in bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w Williams %R(14)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1w) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Calculate 1w ADX(14) for trend filter
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20) + 14  # Need 1w Williams %R(14), volume MA(20), plus ADX smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -20 from below with volume spike AND strong uptrend (ADX>25 and +DI>-DI)
            if (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 and 
                volume_spike[i] and adx_aligned[i] > 25 and plus_di_aligned[i] > minus_di_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from above with volume spike AND strong downtrend (ADX>25 and -DI>+DI)
            elif (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 and 
                  volume_spike[i] and adx_aligned[i] > 25 and minus_di_aligned[i] > plus_di_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) or ADX < 20 (trend weakening)
            if (williams_r_aligned[i] > -50 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) or ADX < 20 (trend weakening)
            if (williams_r_aligned[i] < -50 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1wADX_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0