#!/usr/bin/env python3
name = "12h_TRIX_Breakout_1dTrend_Volume"
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
    
    # Load daily data ONCE for TRIX and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # TRIX (15-period) - momentum oscillator
    # EMA1 = EMA(close, 15)
    ema1 = pd.Series(df_1d['close']).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2 = EMA(EMA1, 15)
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3 = EMA(EMA2, 15)
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Align TRIX to 12h timeframe
    trix_12h = align_htf_to_ltf(prices, df_1d, trix)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(trix_12h[i]) or np.isnan(ema_50_12h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: TRIX crosses above zero in daily uptrend with volume
            if trix_12h[i] > 0 and trix_12h[i-1] <= 0 and ema_50_12h[i] > ema_50_12h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero in daily downtrend with volume
            elif trix_12h[i] < 0 and trix_12h[i-1] >= 0 and ema_50_12h[i] < ema_50_12h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses back below zero or trend reverses
            if trix_12h[i] < 0 or ema_50_12h[i] < ema_50_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses back above zero or trend reverses
            if trix_12h[i] > 0 or ema_50_12h[i] > ema_50_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX zero-cross with daily trend filter and volume confirmation
# - TRIX (15) measures momentum acceleration; zero-cross indicates trend change
# - Long when TRIX crosses above zero in daily uptrend (EMA50 rising)
# - Short when TRIX crosses below zero in daily downtrend (EMA50 falling)
# - Volume confirmation (1.5x average) reduces false signals
# - Exit on TRIX reverse cross or trend reversal
# - Position size 0.25 balances return and risk
# - Works in bull (bullish crosses in uptrend) and bear (bearish crosses in downtrend)
# - Uses 1d timeframe for momentum and trend, 12h for execution timing
# - TRIX is less noisy than MACD, providing cleaner signals with fewer whipsaws
# - Target: 50-100 total trades over 4 years to avoid fee drag
# - Proven pattern: TRIX + volume + trend shows strong performance in ETH per research