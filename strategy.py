#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Load 1w data ONCE before loop for weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter (needs one-bar delay for weekly close)
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla R3/S3
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R3 and S3 levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Wait for volume MA and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike in weekly uptrend
            if close[i] > R3_aligned[i] and vol_spike[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike in weekly downtrend
            elif close[i] < S3_aligned[i] and vol_spike[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below S3 or weekly trend turns down
            if close[i] < S3_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above R3 or weekly trend turns up
            if close[i] > R3_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout on 12h with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 (strong bullish breakout) with volume spike in weekly uptrend.
# Short when price breaks below S3 (strong bearish breakdown) with volume spike in weekly downtrend.
# Uses discrete position size (0.25) to minimize churn. Target 12-37 trades/year.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume spike (>2.0x average) ensures conviction behind the move. R3/S3 are stronger levels than R1/S1.