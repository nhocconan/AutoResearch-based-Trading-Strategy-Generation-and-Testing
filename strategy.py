#!/usr/bin/env python3
name = "6h_RSIStoch_1dTrend"
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
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    rsi_period = 14
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[i-rsi_period+1:i+1])
            avg_loss[i] = np.mean(loss[i-rsi_period+1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic(14,3)
    k_period = 14
    d_period = 3
    lowest_low = np.zeros(n)
    highest_high = np.zeros(n)
    for i in range(k_period-1, n):
        lowest_low[i] = np.min(low[i-k_period+1:i+1])
        highest_high[i] = np.max(high[i-k_period+1:i+1])
    stoch_k = np.where(highest_high != lowest_low, 100 * (close - lowest_low) / (highest_high - lowest_low), 50)
    stoch_d = np.zeros(n)
    for i in range(d_period-1, n):
        if i == d_period-1:
            stoch_d[i] = np.mean(stoch_k[i-d_period+1:i+1])
        else:
            stoch_d[i] = (stoch_d[i-1] * (d_period-1) + stoch_k[i]) / d_period
    
    # 1d trend: EMA(50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.3 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(rsi[i]) or np.isnan(stoch_d[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Entry conditions
        long_entry = (rsi[i] < 30 and stoch_k[i] < 20 and stoch_d[i] < 20 and 
                      close[i] > ema50_1d_aligned[i] and vol_filter)
        short_entry = (rsi[i] > 70 and stoch_k[i] > 80 and stoch_d[i] > 80 and 
                       close[i] < ema50_1d_aligned[i] and vol_filter)
        
        # Exit conditions
        long_exit = (rsi[i] > 70 or stoch_k[i] > 80 or close[i] < ema50_1d_aligned[i])
        short_exit = (rsi[i] < 30 or stoch_k[i] < 20 or close[i] > ema50_1d_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals