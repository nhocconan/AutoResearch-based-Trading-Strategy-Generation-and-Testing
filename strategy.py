#!/usr/bin/env python3
name = "6h_RSI_Stochastic_Combo_1dTrend"
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
    
    # 1d trend filter: EMA(50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) - standard calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / 14
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:15]) if n >= 15 else np.nan
    avg_loss[13] = np.mean(loss[1:15]) if n >= 15 else np.nan
    
    for i in range(14, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator %K(14,3,3)
    lowest_low = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    for i in range(13, n):
        lowest_low[i] = np.min(low[i-13:i+1])
        highest_high[i] = np.max(high[i-13:i+1])
    
    stoch_k_raw = np.divide(close - lowest_low, highest_high - lowest_low, 
                            out=np.full_like(close, np.nan), 
                            where=(highest_high - lowest_low) != 0) * 100
    
    # %K smoothed (3-period SMA)
    stoch_k = np.full(n, np.nan)
    for i in range(2, n):
        if not np.isnan(stoch_k_raw[i-2:i+1]).any():
            stoch_k[i] = np.mean(stoch_k_raw[i-2:i+1])
    
    # %D (3-period SMA of %K)
    stoch_d = np.full(n, np.nan)
    for i in range(2, n):
        if not np.isnan(stoch_k[i-2:i+1]).any():
            stoch_d[i] = np.mean(stoch_k[i-2:i+1])
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        if (np.isnan(rsi[i]) or np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # LONG: RSI < 30 AND Stochastic %K < 20 AND %K crossing above %D AND price above 1d EMA50
        if (rsi[i] < 30 and stoch_k[i] < 20 and 
            stoch_k[i] > stoch_d[i] and 
            close[i] > ema50_1d_aligned[i]):
            signals[i] = 0.25
        # SHORT: RSI > 70 AND Stochastic %K > 80 AND %K crossing below %D AND price below 1d EMA50
        elif (rsi[i] > 70 and stoch_k[i] > 80 and 
              stoch_k[i] < stoch_d[i] and 
              close[i] < ema50_1d_aligned[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals