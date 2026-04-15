#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI with 12h EMA trend filter and volume confirmation
# Uses RSI(14) for mean-reversion signals: long when RSI < 30 and price above 12h EMA50,
# short when RSI > 70 and price below 12h EMA50. Requires volume > 1.5x 20-bar median.
# Designed to work in both bull and bear markets by combining mean-reversion with trend filter.
# Discrete sizing (0.25) limits trade frequency to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: RSI < 30 (oversold), volume spike, price above 12h EMA50
        if (rsi[i] < 30 and volume[i] > vol_threshold[i] and 
            close[i] > ema_12h_aligned[i]):
            signals[i] = 0.25
        
        # Short: RSI > 70 (overbought), volume spike, price below 12h EMA50
        elif (rsi[i] > 70 and volume[i] > vol_threshold[i] and 
              close[i] < ema_12h_aligned[i]):
            signals[i] = -0.25
        
        # Exit: RSI returns to neutral zone (40-60)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and rsi[i] >= 40) or
               (signals[i-1] == -0.25 and rsi[i] <= 60))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_RSI_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0