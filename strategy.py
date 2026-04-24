#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme Reversion with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend.
- Williams %R: Measures overbought/oversold levels (-20 to -80). Extreme readings signal reversals.
- Entry: Long when Williams %R < -90 (extreme oversold) and price > 1d EMA50 (uptrend) with volume spike.
         Short when Williams %R > -10 (extreme overbought) and price < 1d EMA50 (downtrend) with volume spike.
- Exit: When Williams %R reverts to normal range (> -50 for longs, < -50 for shorts) or opposite signal.
- Works in bull via buying panics in uptrend, in bear via selling rallies in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams %R calculation (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 + volume MA + Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Williams %R extreme signals with volume spike and trend filter
            if volume_spike[i]:
                # Long: Extreme oversold (< -90) with uptrend
                if williams_r[i] < -90 and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Extreme overbought (> -10) with downtrend
                elif williams_r[i] > -10 and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R reverts above -50 or trend breaks
            if williams_r[i] > -50 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R reverts below -50 or trend breaks
            if williams_r[i] < -50 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0