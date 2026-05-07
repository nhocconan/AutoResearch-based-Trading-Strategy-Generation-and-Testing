#!/usr/bin/env python3
name = "4h_4H_Camarilla_R1S1_Breakout_12hTrend_Volume"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla levels from previous 1d
    # Daily high, low, close from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1, S1
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Wait for volume MA and EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 with volume spike in uptrend
            if close[i] > R1_aligned[i] and vol_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 with volume spike in downtrend
            elif close[i] < S1_aligned[i] and vol_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S1 or trend turns down
            if close[i] < S1_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R1 or trend turns up
            if close[i] > R1_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above R1 (bullish breakout) with volume spike in 12h uptrend.
# Short when price breaks below S1 (bearish breakdown) with volume spike in 12h downtrend.
# Uses discrete position size (0.25) to minimize churn. Target 20-50 trades/year.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Volume spike (>1.8x average) ensures conviction behind the move.
# Designed for 4h timeframe to target 75-200 total trades over 4 years, avoiding overtrading.