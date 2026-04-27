#!/usr/bin/env python3
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
    
    # Get daily data for 20-period ATR (used for volatility regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily True Range
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Calculate ATR(20) on daily
    atr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 19:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 19 + tr_1d[i]) / 20
    
    # Align ATR(20) to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4-period RSI on 4h closes (for mean reversion signals)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < 3:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else gain[i]
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 3 + gain[i]) / 4
            avg_loss[i] = (avg_loss[i-1] * 3 + loss[i]) / 4
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4-period SMA of RSI for smoothing
    rsi_sma = np.full(n, np.nan)
    for i in range(3, n):
        rsi_sma[i] = np.mean(rsi[i-3:i+1])
    
    # Get weekly data for trend filter: SMA(50) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_1w_50 = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if i < 49:
            sma_1w_50[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            sma_1w_50[i] = np.mean(close_1w[i-49:i+1])
    
    sma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_50)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(3, 50)  # RSI needs 3, weekly SMA needs 50
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_sma[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(sma_1w_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when volatility is elevated
        # ATR(20) > 1.5 * its 50-period SMA (indicates high volatility regime)
        atr_ma_50 = np.full(n, np.nan)
        if i >= 50:
            atr_ma_50[i] = np.mean(atr_1d_aligned[i-50:i])
        
        volatility_filter = (atr_1d_aligned[i] > 1.5 * atr_ma_50[i]) if not np.isnan(atr_ma_50[i]) else False
        
        if not volatility_filter:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price above weekly SMA (bullish bias)
            if (rsi_sma[i] < 30 and 
                price > sma_1w_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) and price below weekly SMA (bearish bias)
            elif (rsi_sma[i] > 70 and 
                  price < sma_1w_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) or price breaks below weekly SMA
            if (rsi_sma[i] > 50 or 
                price < sma_1w_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) or price breaks above weekly SMA
            if (rsi_sma[i] < 50 or 
                price > sma_1w_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_RSI_MeanReversion_VolatilityFilter_WeeklyTrend_v1"
timeframe = "4h"
leverage = 1.0