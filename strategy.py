#!/usr/bin/env python3
"""
6h_ISS_Trend_Reversal
Hypothesis: Combines Intra-Session Sentiment (ISS) from 1d open/close with 6h momentum. 
ISS = (close - open) / (high - low) of prior day. Strong ISS indicates institutional bias. 
Enter long when ISS > 0.2 + 6h RSI(14) < 30 (oversold in uptrend bias). 
Enter short when ISS < -0.2 + 6h RSI(14) > 70 (overbought in downtrend bias). 
Exit when ISS reverses or RSI reaches opposite extreme. 
Works in bull/bear by using daily sentiment as regime filter. Targets 15-30 trades/year.
"""

name = "6h_ISS_Trend_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- Daily ISS (Intra-Session Sentiment) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ISS for each day: (close - open) / (high - low)
    iss_raw = (df_1d['close'].values - df_1d['open'].values) / (df_1d['high'].values - df_1d['low'].values)
    # Avoid division by zero
    iss_raw = np.where((df_1d['high'].values - df_1d['low'].values) == 0, 0, iss_raw)
    iss_1d = iss_raw  # Already daily values
    
    # Align daily ISS to 6h
    iss_6h = align_htf_to_ltf(prices, df_1d, iss_1d)
    
    # --- 6h RSI(14) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # All gains
    rsi = np.where(avg_gain == 0, 0, rsi)    # All losses
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(iss_6h[i]) or np.isnan(rsi[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: strong bullish ISS + RSI oversold
            if iss_6h[i] > 0.2 and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: strong bearish ISS + RSI overbought
            elif iss_6h[i] < -0.2 and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: ISS turns bearish OR RSI overbought
                if iss_6h[i] < 0 or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: ISS turns bullish OR RSI oversold
                if iss_6h[i] > 0 or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals