# 6h_RSI_Trend_Pullback
# Hypothesis: In 60-minute timeframe, buy pullbacks to EMA20 during uptrends (EMA50 rising) when RSI shows oversold conditions (RSI<30), and sell rallies to EMA20 during downtrends (EMA50 falling) when RSI shows overbought conditions (RSI>70). Trend filter uses 1d EMA50 to avoid counter-trend trades. Works in both bull and bear markets by following the higher timeframe trend.
# Entry conditions are strict: RSI extreme + EMA20 touch + trend alignment, targeting ~20-40 trades/year to minimize fee drag.

name = "6h_RSI_Trend_Pullback"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA20 on 6h timeframe
    ema_20 = np.full_like(close, np.nan)
    if len(close) >= 20:
        ema_20[19] = np.mean(close[0:20])
        for i in range(20, len(close)):
            ema_20[i] = (ema_20[i-1] * 19 + close[i]) / 20
    
    # Calculate RSI(14) on 6h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(close) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(close)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 50)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price at EMA20 support, RSI oversold, uptrend (price > 1d EMA50)
            if (low[i] <= ema_20[i] and 
                rsi[i] < 30 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price at EMA20 resistance, RSI overbought, downtrend (price < 1d EMA50)
            elif (high[i] >= ema_20[i] and 
                  rsi[i] > 70 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below EMA20 OR RSI overbought OR trend reversal
            if (close[i] < ema_20[i] or 
                rsi[i] > 70 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above EMA20 OR RSI oversold OR trend reversal
            if (close[i] > ema_20[i] or 
                rsi[i] < 30 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals