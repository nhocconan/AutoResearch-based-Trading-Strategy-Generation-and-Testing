#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA pullback with 4h trend and 1d volume confirmation.
# Long when 1h price pulls back to EMA21 in a 4h uptrend (EMA50 > EMA200) and 1d volume > 1.5x 20 EMA.
# Short when 1h price rallies to EMA21 in a 4h downtrend (EMA50 < EMA200) and 1d volume > 1.5x 20 EMA.
# Uses 4h EMA crossover for trend direction, 1h EMA21 for entry timing, and 1d volume for momentum confirmation.
# Designed for moderate trade frequency (target: 20-40/year) to balance opportunity and fee drag.
# Works in bull markets via 4h uptrend longs and in bear markets via 4h downtrend shorts.
name = "1h_EMA_Pullback_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend (EMA50 and EMA200)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # 4h EMA50 and EMA200
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d volume > 1.5 x 20-period EMA
    vol_ema20_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_condition_1d = df_1d['volume'].values > 1.5 * vol_ema20_1d
    vol_condition_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_condition_1d)
    
    # 1h EMA21 for entry timing
    ema21_1h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(ema21_1h[i]) or np.isnan(vol_condition_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend (EMA50 > EMA200), price at EMA21 support, volume confirmation
            if (ema50_4h_aligned[i] > ema200_4h_aligned[i] and 
                low[i] <= ema21_1h[i] <= high[i] and 
                vol_condition_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend (EMA50 < EMA200), price at EMA21 resistance, volume confirmation
            elif (ema50_4h_aligned[i] < ema200_4h_aligned[i] and 
                  low[i] <= ema21_1h[i] <= high[i] and 
                  vol_condition_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h trend breaks or price moves significantly away from EMA21
            if (ema50_4h_aligned[i] <= ema200_4h_aligned[i] or 
                close[i] > ema21_1h[i] * 1.02):  # 2% above EMA21
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h trend breaks or price moves significantly away from EMA21
            if (ema50_4h_aligned[i] >= ema200_4h_aligned[i] or 
                close[i] < ema21_1h[i] * 0.98):  # 2% below EMA21
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals