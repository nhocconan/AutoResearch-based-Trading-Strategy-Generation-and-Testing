#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-day ATR for volatility
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr_1d = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if i == 13:
            atr_1d[i] = np.mean(tr[:14])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 200-day SMA for trend filter
    sma_200_1d = np.full(len(close_1d), np.nan)
    for i in range(199, len(close_1d)):
        sma_200_1d[i] = np.mean(close_1d[i-199:i+1])
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[:14])
            avg_loss[i] = np.mean(loss[:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 50-day SMA for short-term trend
    sma_50_1d = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        sma_50_1d[i] = np.mean(close_1d[i-49:i+1])
    
    # Align indicators to daily timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup
    start_idx = 199
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(sma_200_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(sma_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        sma_200 = sma_200_1d_aligned[i]
        rsi = rsi_1d_aligned[i]
        sma_50 = sma_50_1d_aligned[i]
        
        # Trend filter: price above/below 200-day SMA
        uptrend = price > sma_200
        downtrend = price < sma_200
        
        if position == 0:
            # Long: Oversold RSI in uptrend with price above 50-day SMA
            if (rsi < 30 and uptrend and price > sma_50):
                signals[i] = size
                position = 1
            # Short: Overbought RSI in downtrend with price below 50-day SMA
            elif (rsi > 70 and downtrend and price < sma_50):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral or trend breaks
            if rsi > 50 or price < sma_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral or trend breaks
            if rsi < 50 or price > sma_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "Daily_RSI_MeanReversion_200SMA_TrendFilter"
timeframe = "1d"
leverage = 1.0