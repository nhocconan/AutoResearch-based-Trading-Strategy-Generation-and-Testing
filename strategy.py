#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h trend filter and session timing
# Uses 4h EMA50 for trend direction, 1h EMA8/EMA21 for entry timing
# Session filter (08-20 UTC) reduces noise in ranging markets
# Discrete position sizing (0.20) controls drawdown and fee drag
# Target: 15-35 trades/year/symbol to avoid fee drag

name = "1h_EMA8_21_4hEMA50_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h EMAs for entry timing
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Pre-compute session hours (08-20 UTC) for efficiency
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if 4h EMA not ready
        if np.isnan(ema_50_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: trade only 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Long: 4h uptrend + 1h bullish crossover
        if (close[i] > ema_50_4h_aligned[i] and  # 4h uptrend
            ema_8[i] > ema_21[i] and             # 1h bullish momentum
            ema_8[i-1] <= ema_21[i-1]):          # bullish crossover
            signals[i] = 0.20
            
        # Short: 4h downtrend + 1h bearish crossover
        elif (close[i] < ema_50_4h_aligned[i] and  # 4h downtrend
              ema_8[i] < ema_21[i] and             # 1h bearish momentum
              ema_8[i-1] >= ema_21[i-1]):          # bearish crossover
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals