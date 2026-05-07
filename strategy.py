#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_12hEMA25_VolumeSpike"
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
    
    # Load 1d data ONCE before loop for Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h EMA(25) for trend filter
    ema_25_12h = pd.Series(df_12h['close']).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # Camarilla levels from previous day (daily high/low/close)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla formula: (H+L+C)/3 is pivot, then R1/S1 etc.
    camarilla_pivot = (daily_high + daily_low + daily_close) / 3
    camarilla_range = daily_high - daily_low
    r1 = camarilla_pivot + camarilla_range * 1.1 / 12
    s1 = camarilla_pivot - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: 20-period average (approx 3.3 days of 4h bars)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(25, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_25_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume spike and 12h uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            uptrend = ema_25_12h_aligned[i] > ema_25_12h_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume spike and 12h downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout with 12h EMA trend and volume spike
# - Camarilla R1/S1 act as intraday support/resistance levels derived from prior day's range
# - Break above S1 with volume spike (2x average) in 12h uptrend = long
# - Break below R1 with volume spike in 12h downtrend = short
# - Volume spike confirms institutional participation and reduces false breakouts
# - Works in bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to the breakout level (S1/R1) or volume weakens
# - Position size 0.25 targets ~30-50 trades/year, avoiding excessive fee drag
# - Uses 12h EMA for smoother trend filter less prone to whipsaw than shorter EMAs
# - Camarilla levels from daily data provide institutional reference points that work across regimes