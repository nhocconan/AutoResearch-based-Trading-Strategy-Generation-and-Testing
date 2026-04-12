#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_camarilla_pivot_volume_v1
# Uses Camarilla pivot levels from 12h timeframe for entry/exit signals on 4h chart.
# Long when price touches S3 level with volume confirmation, short when touches R3 level.
# Volume confirmation requires current volume > 1.5x average volume of last 20 periods.
# Includes volatility filter using ATR to avoid whipsaws in low volatility conditions.
# Designed to work in both bull and bear markets by capturing mean reversion at extreme levels.
name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 12h timeframe
    # Using previous period's high, low, close
    ph = df_12h['high'].shift(1).values  # previous high
    pl = df_12h['low'].shift(1).values   # previous low
    pc = df_12h['close'].shift(1).values # previous close
    
    # Calculate pivot and ranges
    pivot = (ph + pl + pc) / 3
    range_ = ph - pl
    
    # Camarilla levels
    s3 = pc - (range_ * 1.1 / 2)
    s2 = pc - (range_ * 1.1 / 4)
    s1 = pc - (range_ * 1.1 / 6)
    r1 = pc + (range_ * 1.1 / 6)
    r2 = pc + (range_ * 1.1 / 4)
    r3 = pc + (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    
    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Volatility filter using ATR to avoid low volatility whipsaws
    # Calculate ATR(14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
    # Only trade when ATR is above its 50-period average (avoid low volatility)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=1).mean().values
    volatility_filter = atr >= (atr_ma * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(volume_confirm[i]) or np.isnan(volatility_filter[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long signal: price touches or goes below S3 with volume and volatility confirmation
        long_signal = (low[i] <= s3_aligned[i]) and volume_confirm[i] and volatility_filter[i]
        
        # Short signal: price touches or goes above R3 with volume and volatility confirmation
        short_signal = (high[i] >= r3_aligned[i]) and volume_confirm[i] and volatility_filter[i]
        
        # Exit conditions: opposite touch or reversal
        exit_long = (high[i] >= s3_aligned[i])  # exit long when price moves back above S3
        exit_short = (low[i] <= r3_aligned[i])  # exit short when price moves back below R3
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals