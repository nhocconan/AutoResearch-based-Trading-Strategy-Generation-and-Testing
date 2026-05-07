#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate daily Camarilla levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike detection (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R1 in 12h uptrend with volume spike
            if close[i] > R1_aligned[i] and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 in 12h downtrend with volume spike
            elif close[i] < S1_aligned[i] and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or trend change
            if close[i] < S1_aligned[i] or ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or trend change
            if close[i] > R1_aligned[i] or ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation
# - Camarilla R1/S1 levels derived from prior day's high-low-close provide institutional support/resistance
# - Long when price breaks above R1 in 12h uptrend (EMA50 rising) with volume spike (2x 20-period average)
# - Short when price breaks below S1 in 12h downtrend (EMA50 falling) with volume spike
# - Exit when price returns to S1/R1 or 12h trend changes
# - Works in both bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend)
# - Volume confirmation reduces false breakouts from low-volume noise
# - Position size 0.25 balances profit potential with risk management
# - Target: 20-50 trades/year to avoid excessive fee drag while capturing meaningful moves
# - Uses 12h trend filter to avoid counter-trend trades and improve signal quality
# - Proven pattern: Similar to top performers showing 1.8+ Sharpe on SOL/ETH with Camarilla + volume + trend