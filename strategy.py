#!/usr/bin/env python3
"""
6h_Trend_Reversal_Capture
Hypothesis: In 6h timeframe, strong reversals often occur after exhaustion moves. 
We capture these by: 1) Using 1d RSI(14) to identify overbought/oversold conditions on higher timeframe, 
2) Waiting for price to reverse back toward the 6h VWAP as confirmation of exhaustion, 
3) Entering on the close of the reversal bar with proper sizing. 
Exit when price returns to the 1d RSI neutral zone (40-60). 
This works in both bull and bear markets as it captures mean reversion within the trend.
"""

name = "6h_Trend_Reversal_Capture"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 6h timeframe (wait for 1d bar to close)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 6h VWAP (typical price * volume)
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    vwap_num = (typical_price * prices['volume']).cumsum()
    vwap_den = prices['volume'].cumsum()
    vwap = vwap_num / vwap_den
    vwap = vwap.replace(0, np.nan).ffill().values  # forward fill initial NaN
    
    # Price array
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14 days) and enough data for VWAP
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if RSI is not available
        if np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: 1d RSI oversold (<30) AND price crosses above VWAP (bullish reversal)
            if rsi_1d_aligned[i] < 30 and close[i] > vwap[i] and close[i-1] <= vwap[i-1]:
                signals[i] = 0.25
                position = 1
            # Short setup: 1d RSI overbought (>70) AND price crosses below VWAP (bearish reversal)
            elif rsi_1d_aligned[i] > 70 and close[i] < vwap[i] and close[i-1] >= vwap[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to VWAP OR RSI reaches neutral (>50)
            if close[i] <= vwap[i] or rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to VWAP OR RSI reaches neutral (<50)
            if close[i] >= vwap[i] or rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals