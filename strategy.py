#!/usr/bin/env python3
# Hypothesis: 1h mean reversion with 4h trend filter and volume confirmation
# Long when price pulls back to 20-period EMA in 4h uptrend with volume above average
# Short when price rallies to 20-period EMA in 4h downtrend with volume above average
# Exit when price crosses 10-period EMA in opposite direction
# Uses 4h trend to filter direction and 1h for precise entry timing
# Designed to work in both bull (buy dips) and bear (sell rallies) markets
# Target: 60-150 total trades over 4 years = 15-37/year for 1h with size 0.20

name = "1h_EMA_Pullback_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA20 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h EMA10 and EMA20 for entry/exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.2x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_10[i]) or np.isnan(ema_20[i]) or np.isnan(vol_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price near 20 EMA in 4h uptrend with volume
            if (close[i] <= ema_20[i] * 1.005 and  # within 0.5% above EMA20
                close[i] >= ema_20[i] * 0.995 and  # within 0.5% below EMA20
                ema_4h_aligned[i] > ema_4h_aligned[max(i-1, start_idx)] and  # 4h EMA rising
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price near 20 EMA in 4h downtrend with volume
            elif (close[i] <= ema_20[i] * 1.005 and
                  close[i] >= ema_20[i] * 0.995 and
                  ema_4h_aligned[i] < ema_4h_aligned[max(i-1, start_idx)] and  # 4h EMA falling
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 10 EMA
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above 10 EMA
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals