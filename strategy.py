#!/usr/bin/env python3
name = "6h_ElderRay_BullPower_BearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 13-period EMA for Elder Ray (standard period)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 with volume spike and 1d uptrend
            if (bull_power[i] > 0 and 
                volume[i] > vol_ma[i] * 1.5 and 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 with volume spike and 1d downtrend
            elif (bear_power[i] < 0 and 
                  volume[i] > vol_ma[i] * 1.5 and 
                  close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative or trend changes
            if (bull_power[i] <= 0 or 
                close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive or trend changes
            if (bear_power[i] >= 0 or 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA(34) trend filter and volume confirmation.
# Elder Ray measures the power of bulls (high - EMA13) and bears (low - EMA13) relative to trend.
# Bull Power > 0 indicates bulls are stronger than the trend; Bear Power < 0 indicates bears are stronger.
# Combined with 1d trend filter to ensure we trade with the higher timeframe direction.
# Volume confirmation validates the strength of the move.
# Works in bull markets (buying when Bull Power > 0) and bear markets (selling when Bear Power < 0).
# Position size 0.25 balances risk and keeps trade frequency manageable (target: 50-150 trades over 4 years).