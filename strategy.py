#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d RSI for momentum filter
    delta = close_1d.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi1d = 100 - (100 / (1 + rs))
    rsi1d = rsi1d.fillna(50).values
    rsi1d_aligned = align_htf_to_ltf(prices, df_1d, rsi1d)
    
    # 4h Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_val = df_1d['close'].values
    
    # Camarilla R1 and S1
    camarilla_r1 = close_1d_val + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d_val - (high_1d - low_1d) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 4h volume spike: > 1.5x 24-period average (24 * 4h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # Wait for volume MA and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(rsi1d_aligned[i]) or \
           np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R1, bullish trend, volume spike
            if close[i] > camarilla_r1_aligned[i] and ema34_1d_aligned[i] > 0 and rsi1d_aligned[i] > 50 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1, bearish trend, volume spike
            elif close[i] < camarilla_s1_aligned[i] and ema34_1d_aligned[i] > 0 and rsi1d_aligned[i] < 50 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below Camarilla S1 or bearish momentum
            if close[i] < camarilla_s1_aligned[i] or rsi1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above Camarilla R1 or bullish momentum
            if close[i] > camarilla_r1_aligned[i] or rsi1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and 1d RSI momentum filter.
# Long when price breaks above Camarilla R1, 1d trend is bullish (EMA34 > 0), RSI > 50, and volume spike confirms.
# Short when price breaks below Camarilla S1, 1d trend is bullish (EMA34 > 0), RSI < 50, and volume spike confirms.
# Uses 1d timeframe for trend/momentum to avoid whipsaws, 4h for entry timing.
# Volume spike (>1.5x average) ensures conviction. Discrete 0.25 position size limits risk.
# Works in bull markets (trend + momentum) and bear markets (reverse criteria).
# Target: 20-50 trades/year to minimize fee drag while capturing sustained moves.