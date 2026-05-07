#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d RSI for momentum filter
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi1d = 100 - (100 / (1 + rs))
    rsi1d = rsi1d.fillna(50).values  # Neutral when undefined
    rsi1d_aligned = align_htf_to_ltf(prices, df_1d, rsi1d)
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1
    r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h volume spike: > 1.5x 24-period average (6 days of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # Wait for volume MA and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(rsi1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, bullish trend (price > EMA34), bullish momentum (RSI > 50), volume spike
            if close[i] > r1_aligned[i] and close[i] > ema34_1d_aligned[i] and rsi1d_aligned[i] > 50 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, bearish trend (price < EMA34), bearish momentum (RSI < 50), volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema34_1d_aligned[i] and rsi1d_aligned[i] < 50 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below S1 or bearish momentum (RSI < 40)
            if close[i] < s1_aligned[i] or rsi1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above R1 or bullish momentum (RSI > 60)
            if close[i] > r1_aligned[i] or rsi1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and 1d RSI momentum filter.
# Long when price breaks above Camarilla R1, price > 1d EMA34 (bullish trend), RSI > 50 (bullish momentum), and volume spike confirms.
# Short when price breaks below Camarilla S1, price < 1d EMA34 (bearish trend), RSI < 50 (bearish momentum), and volume spike confirms.
# Uses 1d timeframe for trend/momentum to avoid whipsaws, 4h for entry timing.
# Volume spike (>1.5x average) ensures conviction. Discrete 0.25 position size limits risk.
# Works in bull markets (breakout + trend + momentum) and bear markets (reverse criteria).
# Target: 20-50 trades/year to minimize fee drag while capturing sustained moves.