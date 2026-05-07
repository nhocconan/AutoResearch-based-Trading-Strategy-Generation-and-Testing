# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume - 12h timeframe with 1d trend filter
# Targets 50-150 trades over 4 years (12-37/year) to avoid fee drag
# Uses Camarilla R3/S3 levels from previous day with volume spike confirmation
# Trend filter: 1d EMA34 ensures we trade with higher timeframe trend
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend)
# Discrete position sizing (0.25) minimizes churn from frequent signal changes

#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Load 1d data ONCE before loop for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels (more significant than R1/S1)
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: > 2.0x 30-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 35)  # Wait for volume MA and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R3 with volume spike in uptrend
            if close[i] > R3_aligned[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 with volume spike in downtrend
            elif close[i] < S3_aligned[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S3 or trend turns down
            if close[i] < S3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R3 or trend turns up
            if close[i] > R3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout on 12h with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 (bullish breakout) with volume spike in 1d uptrend.
# Short when price breaks below S3 (bearish breakdown) with volume spike in 1d downtrend.
# Uses discrete position size (0.25) to minimize churn. Target 12-37 trades/year.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Volume spike (>2.0x average) ensures conviction behind the move.
# R3/S3 levels are more significant than R1/S1, reducing false breakouts.