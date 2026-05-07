#3/usd/2025-06-08
#!/usr/bin/env python3
name = "6h_RSI_20_EMA50_Crossover_1dTrend_Volume"
timeframe = "6h"
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
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h RSI(20)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[20] = np.mean(gain[1:21])
    avg_loss[20] = np.mean(loss[1:21])
    
    for i in range(21, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 19 + gain[i]) / 20
        avg_loss[i] = (avg_loss[i-1] * 19 + loss[i]) / 20
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 6h EMA50
    ema50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 6h volume spike: > 2x 20-period average
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_6h = volume > 2.0 * vol_ma_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema50_6h[i]) or np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI > 50, price above EMA50, bullish trend (price > EMA50_1d), volume spike
            if (rsi[i] > 50 and close[i] > ema50_6h[i] and 
                close[i] > ema50_1d_aligned[i] and vol_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 50, price below EMA50, bearish trend (price < EMA50_1d), volume spike
            elif (rsi[i] < 50 and close[i] < ema50_6h[i] and 
                  close[i] < ema50_1d_aligned[i] and vol_spike_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI < 40 or price below EMA50
            if rsi[i] < 40 or close[i] < ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI > 60 or price above EMA50
            if rsi[i] > 60 or close[i] > ema50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: RSI(20) > 50 indicates bullish momentum, < 50 bearish.
# Combined with EMA50 crossover and 1d trend filter for direction.
# Volume spike confirms momentum. Works in bull/bear by following 1d trend.
# Target: 20-40 trades/year to minimize fee drift. Position size 0.25 limits risk.