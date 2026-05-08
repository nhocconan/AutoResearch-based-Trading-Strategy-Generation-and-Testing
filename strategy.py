#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 123 reversal pattern with daily trend filter and volume confirmation
# Long when close > high of previous 2 bars (123 buy), daily EMA50 up, volume spike
# Short when close < low of previous 2 bars (123 sell), daily EMA50 down, volume spike
# 123 pattern captures short-term momentum shifts; daily trend filters for higher timeframe direction
# Volume spike confirms institutional participation; avoids false signals
# Targets 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost

name = "4h_123_Reversal_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: 123 buy pattern (close > high of previous 2 bars), daily uptrend, volume spike
            if i >= 2 and close[i] > high[i-1] and close[i] > high[i-2] and ema50_1d_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: 123 sell pattern (close < low of previous 2 bars), daily downtrend, volume spike
            elif i >= 2 and close[i] < low[i-1] and close[i] < low[i-2] and ema50_1d_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close < low of previous bar or daily trend turns down
            if i >= 1 and close[i] < low[i-1] or ema50_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close > high of previous bar or daily trend turns up
            if i >= 1 and close[i] > high[i-1] or ema50_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals