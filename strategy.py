#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_ema_crossover_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for EMA and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMAs for trend filter (50 and 200)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMAs to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 6h EMA crossover (9 and 21) for entry signals
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 6h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above both daily EMAs for long, below both for short
        uptrend = close[i] > ema_50_1d_aligned[i] and close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i] and close[i] < ema_200_1d_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # EMA crossover signals
        ema_cross_up = ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]
        ema_cross_down = ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]
        
        # Trading logic
        if ema_cross_up and uptrend and volume_confirmed and position != 1:
            position = 1
            signals[i] = 0.25
        elif ema_cross_down and downtrend and volume_confirmed and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (ema_9[i] < ema_21[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (ema_9[i] > ema_21[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h EMA crossover strategy with daily trend filter and volume confirmation.
# Uses 9/21 EMA crossover on 6h timeframe for entry signals.
# Only takes longs when price is above both daily 50 and 200 EMA (uptrend).
# Only takes shorts when price is below both daily 50 and 200 EMA (downtrend).
# Requires volume > 1.3x 20-period average to confirm conviction.
# Exits when EMA crossover reverses or trend filter fails.
# Designed to work in both bull and bear markets by following the daily trend.
# Position size: 0.25 to balance risk and return, limiting drawdown.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.