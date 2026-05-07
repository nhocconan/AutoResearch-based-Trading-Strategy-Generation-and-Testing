#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1H momentum with 4H trend and volume confirmation
# Uses 4H EMA for trend direction, 1H RSI for momentum, and volume spike for confirmation
# Designed for low trade frequency (15-37/year) to minimize fee drag
# Works in both bull and bear markets by following 4H trend
timeframe = "1h"
name = "1H_Momentum_4HTrend_Volume"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4H EMA trend filter (21-period)
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1H RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 4H EMA, RSI > 50, volume spike
            if (close[i] > ema_4h_aligned[i] and 
                rsi[i] > 50 and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below 4H EMA, RSI < 50, volume spike
            elif (close[i] < ema_4h_aligned[i] and 
                  rsi[i] < 50 and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price below 4H EMA or RSI < 40
            if (close[i] < ema_4h_aligned[i] or rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price above 4H EMA or RSI > 60
            if (close[i] > ema_4h_aligned[i] or rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals