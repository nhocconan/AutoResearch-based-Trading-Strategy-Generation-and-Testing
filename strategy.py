#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_RSI_Momentum_Pullback"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for RSI calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 12h close
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 50-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 50 for 1d EMA and 20 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi = rsi_12h_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: RSI > 55 (momentum) AND price > 1d EMA50 (uptrend) AND volume spike
            if rsi > 55 and close[i] > ema_50 and vol > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI < 45 (momentum) AND price < 1d EMA50 (downtrend) AND volume spike
            elif rsi < 45 and close[i] < ema_50 and vol > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI < 40 (loss of momentum) OR price < 1d EMA50 (trend reversal)
            if rsi < 40 or close[i] < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI > 60 (loss of momentum) OR price > 1d EMA50 (trend reversal)
            if rsi > 60 or close[i] > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals