#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d RSI(14) for overbought/oversold
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1w close for trend filter
    close_1w = df_1w['close'].values
    sma_10_1w = pd.Series(close_1w).rolling(window=10, min_periods=10).mean().values
    sma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_10_1w)
    
    # Calculate 6-hour ATR(14) for volatility filter and stoploss
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_1d_aligned[i]
        sma_10w_val = sma_10_1w_aligned[i]
        atr_val = atr_6h[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(sma_10w_val) or 
            np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) with weekly uptrend (price > 10w SMA)
            if rsi_val < 30 and close_val > sma_10w_val and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) with weekly downtrend (price < 10w SMA)
            elif rsi_val > 70 and close_val < sma_10w_val and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought (>70) or ATR-based stop
            if rsi_val > 70 or close_val < prices['high'].iloc[i] - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) or ATR-based stop
            if rsi_val < 30 or close_val > prices['low'].iloc[i] + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_1dRSI_1wTrend_ATRFilter_V1
# Uses 1-day RSI(14) for overbought/oversold signals
# Filters by 1-week trend (price vs 10-period SMA)
# Enters long when RSI < 30 and weekly uptrend (price > 10w SMA)
# Enters short when RSI > 70 and weekly downtrend (price < 10w SMA)
# Uses 6h ATR(14) for volatility filter and 2*ATR stoploss
# Designed for 6h timeframe with ~12-37 trades/year
name = "6h_1dRSI_1wTrend_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0