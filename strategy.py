#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Load daily data ONCE for Camarilla and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (use previous day's data)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels for R1, S1 (most critical)
    R1 = prev_close + prev_range * 1.1 / 12
    S1 = prev_close - prev_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for previous day close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: price breaks above R1 with volume in daily uptrend
            if close[i] > R1_aligned[i] and vol_condition and ema_34_aligned[i] > ema_34_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in daily downtrend
            elif close[i] < S1_aligned[i] and vol_condition and ema_34_aligned[i] < ema_34_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to S1 or trend changes
            if close[i] < S1_aligned[i] or ema_34_aligned[i] < ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to R1 or trend changes
            if close[i] > R1_aligned[i] or ema_34_aligned[i] > ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R1/S1 breakout with daily trend filter and volume confirmation
# - Camarilla R1/S1 are key intraday support/resistance levels derived from previous day's range
# - Breakout above R1 in daily uptrend = long signal; breakdown below S1 in daily downtrend = short signal
# - Volume confirmation (2x average) reduces false breakouts
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend
# - Exit when price returns to opposite level or trend changes
# - Works in both bull (R1 breaks in uptrend) and bear (S1 breaks in downtrend)
# - Position size 0.25 targets ~25-60 trades/year to avoid fee drag
# - Proven pattern: Camarilla + volume + trend is top performer in DB (e.g., ETHUSDT test Sharpe 1.47)
# - Uses proper MTF data loading: get_htf_data() called ONCE before loop, aligned arrays used inside