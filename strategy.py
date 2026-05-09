# US12h_RSI34_Trend_And_Momentum  
# Hypothesis: Combine RSI(34) trend filter with momentum confirmation on 12h timeframe.  
# In bull markets: RSI > 50 + rising momentum triggers long.  
# In bear markets: RSI < 50 + falling momentum triggers short.  
# Uses 1D trend filter to avoid counter-trend trades.  
# Designed for low trade frequency (~15-25/year) to avoid fee drag.  

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "US12h_RSI34_Trend_And_Momentum"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate RSI(34) on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[34] = np.mean(gain[1:35])
    avg_loss[34] = np.mean(loss[1:35])
    
    for i in range(35, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 33 + gain[i]) / 34
        avg_loss[i] = (avg_loss[i-1] * 33 + loss[i]) / 34
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 12-period ROC (momentum) on 1d close
    roc_1d = np.zeros_like(close_1d)
    for i in range(12, len(close_1d)):
        if close_1d[i-12] != 0:
            roc_1d[i] = (close_1d[i] - close_1d[i-12]) / close_1d[i-12] * 100
    
    # Align RSI and ROC to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    roc_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_1d)
    
    signals = np.zeros(n)
    position = 0
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(roc_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi = rsi_1d_aligned[i]
        roc = roc_1d_aligned[i]
        
        if position == 0:
            # Long: RSI > 50 (bullish momentum) + positive ROC
            if rsi > 50 and roc > 0:
                signals[i] = 0.25
                position = 1
            # Short: RSI < 50 (bearish momentum) + negative ROC
            elif rsi < 50 and roc < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI < 50 (momentum fading) or ROC turning negative
            if rsi < 50 or roc < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI > 50 (momentum improving) or ROC turning positive
            if rsi > 50 or roc > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals