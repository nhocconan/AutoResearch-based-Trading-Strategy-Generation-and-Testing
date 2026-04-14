#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h EMA crossover with 1d trend filter and volume confirmation
# Targets: 20-40 trades/year by requiring bullish/bearish EMA crossover during strong trend
# Logic: Long when EMA9 crosses above EMA21, price > 1d EMA50, volume > 1.5x average
#        Short when EMA9 crosses below EMA21, price < 1d EMA50, volume > 1.5x average
#        Uses EMA crossover for momentum, 1d EMA50 for trend filter, volume for confirmation
# Position size: 0.25 to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # EMA9 and EMA21 for crossover
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned daily EMA50
        ema_50_i = align_htf_to_ltf(prices, df_1d, ema_50_1d)[i]
        
        if np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema_50_i) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: EMA9 crosses above EMA21, uptrend, volume confirmation
        if position == 0 and ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and close[i] > ema_50_i and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: EMA9 crosses below EMA21, downtrend, volume confirmation
        elif position == 0 and ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] and close[i] < ema_50_i and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Opposite crossover
        elif position != 0:
            if position == 1 and ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_EMACrossover_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0