#!/usr/bin/env python3
# 4h_1d_ema_rsi_v1
# Strategy: 4h EMA crossover with 1d RSI filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: EMA crossover (21/55) captures trends, while 1d RSI (40/60) filters out counter-trend entries. This reduces false signals and improves win rate. Designed for low trade frequency (~25-40/year) to avoid fee drag. Works in bull/bear via trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_rsi_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d RSI(14) for trend filter
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    
    # 4h EMA21 and EMA55
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(55, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21[i]) or np.isnan(ema_55[i]) or 
            np.isnan(rsi_14_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # EMA crossover signals
        ema_cross_up = ema_21[i] > ema_55[i] and ema_21[i-1] <= ema_55[i-1]
        ema_cross_down = ema_21[i] < ema_55[i] and ema_21[i-1] >= ema_55[i-1]
        
        # 1d RSI filter: RSI > 50 = bullish bias, RSI < 50 = bearish bias
        rsi_bullish = rsi_14_aligned[i] > 50
        rsi_bearish = rsi_14_aligned[i] < 50
        
        # Entry conditions
        # Long: EMA21 crosses above EMA55 AND RSI > 50
        if ema_cross_up and rsi_bullish and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: EMA21 crosses below EMA55 AND RSI < 50
        elif ema_cross_down and rsi_bearish and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite EMA crossover
        elif position == 1 and ema_cross_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and ema_cross_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals