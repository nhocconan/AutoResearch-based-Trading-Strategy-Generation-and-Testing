#!/usr/bin/env python3
name = "4h_Trix_Signal_Line_Cross_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for TRIX calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # TRIX on daily close: EMA(EMA(EMA(close, 12), 12), 12) - 1-period ROC
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix_raw.fillna(0).values
    
    # TRIX signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX and signal line to 4h timeframe
    trix_4h = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_4h = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (1.8x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(trix_4h[i]) or np.isnan(trix_signal_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.8
        trix_cross_up = trix_4h[i] > trix_signal_4h[i] and trix_4h[i-1] <= trix_signal_4h[i-1]
        trix_cross_down = trix_4h[i] < trix_signal_4h[i] and trix_4h[i-1] >= trix_signal_4h[i-1]
        
        if position == 0:
            # Long: TRIX crosses above signal line in daily uptrend with volume
            if trix_cross_up and ema_50_4h[i] > ema_50_4h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line in daily downtrend with volume
            elif trix_cross_down and ema_50_4h[i] < ema_50_4h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below signal line or trend reverses
            if trix_cross_down or ema_50_4h[i] < ema_50_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above signal line or trend reverses
            if trix_cross_up or ema_50_4h[i] > ema_50_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX (triple exponential average) signal line crossovers with daily trend filter and volume confirmation
# - TRIX filters out insignificant price movements and highlights significant trends
# - Signal line crossovers provide timely entry/exit signals
# - Daily EMA50 trend filter ensures trades align with higher timeframe momentum
# - Volume confirmation (1.8x average) reduces false signals from low-volume breakouts
# - Works in bull markets (buy signals in uptrends) and bear markets (sell signals in downtrends)
# - Position size 0.25 balances return potential with risk management
# - Targets ~40-60 trades/year to minimize fee drag while capturing meaningful trends
# - Uses 1d timeframe for TRIX calculation and trend filtering, 4h for execution timing