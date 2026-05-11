#!/usr/bin/env python3
name = "12h_FlippingPoint_RSI_BB_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # RSI(14) on 12H
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20, 2) on 12H
    sma20 = np.zeros(n)
    std20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            sma20[i] = np.mean(close[:i+1])
            std20[i] = np.std(close[:i+1]) if i > 0 else 0
        else:
            sma20[i] = np.mean(close[i-19:i+1])
            std20[i] = np.std(close[i-19:i+1])
    
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Weekly trend: price > SMA50 on 1W
    sma50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(sma50_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Flipping point: RSI crosses 50 with Bollinger Band support/resistance
        rsi_prev = rsi[i-1] if i > 0 else 50
        rsi_cross_up = rsi_prev < 50 and rsi[i] >= 50
        rsi_cross_down = rsi_prev > 50 and rsi[i] <= 50
        
        bb_support = close[i] <= lower_bb[i] * 1.02  # near or below lower BB
        bb_resistance = close[i] >= upper_bb[i] * 0.98  # near or above upper BB
        
        weekly_uptrend = close[i] > sma50_1w_aligned[i]
        weekly_downtrend = close[i] < sma50_1w_aligned[i]
        
        if position == 0:
            # Long: RSI crosses above 50, near lower BB, weekly uptrend, volume surge
            if (rsi_cross_up and bb_support and weekly_uptrend and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 50, near upper BB, weekly downtrend, volume surge
            elif (rsi_cross_down and bb_resistance and weekly_downtrend and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses below 50 OR price crosses above upper BB OR weekly trend fails
            if (rsi_cross_down or close[i] >= upper_bb[i] or not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses above 50 OR price crosses below lower BB OR weekly trend fails
            if (rsi_cross_up or close[i] <= lower_bb[i] or not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals