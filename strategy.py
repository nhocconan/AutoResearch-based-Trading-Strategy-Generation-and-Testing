#!/usr/bin/env python3
name = "6h_RSI_Divergence_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI(14) for momentum filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 6h RSI(14) for divergence detection
    delta_6h = np.diff(close, prepend=close[0])
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    avg_gain_6h = pd.Series(gain_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_6h = pd.Series(loss_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_6h = avg_gain_6h / (avg_loss_6h + 1e-10)
    rsi_6h = 100 - (100 / (1 + rs_6h))
    
    # Calculate 6h EMA(50) for trend filter
    ema_50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(rsi_6h[i]) or 
            np.isnan(ema_50_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (i >= 2 and 
                close[i] < close[i-2] and 
                rsi_6h[i] > rsi_6h[i-2] and
                rsi_1d_aligned[i] > 50 and  # 1d momentum bullish
                close[i] > ema_50_6h[i]):    # 6h trend filter
                signals[i] = 0.25
                position = 1
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (i >= 2 and 
                  close[i] > close[i-2] and 
                  rsi_6h[i] < rsi_6h[i-2] and
                  rsi_1d_aligned[i] < 50 and  # 1d momentum bearish
                  close[i] < ema_50_6h[i]):    # 6h trend filter
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or trend breaks down
            if (rsi_6h[i] > 70 or 
                close[i] < ema_50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold or trend breaks up
            if (rsi_6h[i] < 30 or 
                close[i] > ema_50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals