# -*- coding: utf-8 -*-
# -*- mode: python -*-
#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day momentum with weekly trend filter
# Uses 1d RSI(14) and price position relative to 20-day SMA for mean-reversion entries
# Weekly trend filter (price above/below 20-week SMA) prevents counter-trend trades
# Volume confirmation ensures sufficient participation
# Target: 30-100 total trades over 4 years (7-25/year) with disciplined entries
# Works in bull markets via trend-following bias and in bear markets via mean-reversion within trend
name = "1d_RSI_SMA_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d RSI(14) for mean-reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = gain[:14].mean()
    avg_loss[13] = loss[:14].mean()
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d SMA(20) for mean-reversion reference
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter: price vs 20-week SMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    sma_20_1w = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).mean().values
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(sma_20[i]) or 
            np.isnan(sma_20_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + price below SMA20 + above weekly trend + volume
            if (rsi[i] < 30 and 
                close[i] < sma_20[i] and 
                close[i] > sma_20_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + price above SMA20 + below weekly trend + volume
            elif (rsi[i] > 70 and 
                  close[i] > sma_20[i] and 
                  close[i] < sma_20_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI overbought (>70) or price crosses above SMA20
            if (rsi[i] > 70) or (close[i] > sma_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI oversold (<30) or price crosses below SMA20
            if (rsi[i] < 30) or (close[i] < sma_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals