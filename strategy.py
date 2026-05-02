#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume regime
# Uses 6h timeframe to balance signal quality and trade frequency (target: 50-150 trades over 4 years)
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 12h EMA50 provides trend filter for higher probability entries
# Volume regime filter (current volume > 1.5x 20-period MA) confirms participation
# Works in bull markets via bull power + trend alignment and bear markets via bear power + trend alignment
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "6h_ElderRay_12hEMA50_VolumeRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull power: high minus EMA13
    bear_power = low - ema_13   # Bear power: low minus EMA13
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h volume regime (current volume > 1.5x 20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: bull power > 0 AND price > 12h EMA50 (bullish trend) AND volume regime
            if (bull_power[i] > 0 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bear power < 0 AND price < 12h EMA50 (bearish trend) AND volume regime
            elif (bear_power[i] < 0 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bull power <= 0 OR price < 12h EMA50 (trend change)
            if bull_power[i] <= 0 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bear power >= 0 OR price > 12h EMA50 (trend change)
            if bear_power[i] >= 0 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals