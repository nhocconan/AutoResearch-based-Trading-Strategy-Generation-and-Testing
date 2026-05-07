#/usr/bin/env python3
name = "12h_Donchian_20_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Load daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-day)
    upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Volume spike: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Wait for Donchian and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume spike in weekly uptrend
            if close[i] > upper_aligned[i] and vol_spike[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with volume spike in weekly downtrend
            elif close[i] < lower_aligned[i] and vol_spike[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below lower Donchian or weekly trend turns down
            if close[i] < lower_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above upper Donchian or weekly trend turns up
            if close[i] > upper_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout on 12h with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above 20-day high with volume spike in weekly uptrend.
# Short when price breaks below 20-day low with volume spike in weekly downtrend.
# Uses discrete position size (0.25) to minimize churn. Target 12-37 trades/year.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Volume spike (>1.5x average) ensures conviction behind the move.
# Designed for 12h timeframe to target 50-150 total trades over 4 years, avoiding overtrading.