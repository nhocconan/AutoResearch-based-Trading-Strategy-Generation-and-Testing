#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter and volume confirmation.
Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA50 (uptrend bias) AND volume > 1.3x 20-period average.
Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA50 (downtrend bias) AND volume > 1.3x 20-period average.
Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
Uses 1d HTF for EMA50 trend bias to avoid counter-trend trades in strong trends. Target: 50-150 total trades over 4 years (12-37/year).
Williams %R captures short-term overbought/oversold conditions; EMA50 filter ensures we trade with the higher timeframe trend.
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
    
    # Calculate 1d EMA50 for trend bias (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50 + 49, 20)  # williams_r (14), ema calculation (50+49), vol_ma (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_bias = ema_50_1d_aligned[i]
        wr = williams_r[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Oversold AND uptrend bias AND volume confirmation
            if wr < -80 and price > ema_bias and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Overbought AND downtrend bias AND volume confirmation
            elif wr > -20 and price < ema_bias and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when Williams %R crosses back above -50 (long) or below -50 (short)
            if position == 1 and wr > -50:
                exit_signal = True
            elif position == -1 and wr < -50:
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_1dEMA50_TrendBias_VolumeConfirmation_WR50Exit"
timeframe = "6h"
leverage = 1.0