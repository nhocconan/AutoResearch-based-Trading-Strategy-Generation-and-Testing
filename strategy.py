#!/usr/bin/env python3
"""
6h_1d_RSI_Reversal_v1
Hypothesis: Mean reversion on 6b using daily RSI extremes (overbought/oversold) with volume confirmation.
Enter long when daily RSI < 30 and 6h price closes above 6h VWAP with volume > 1.5x average.
Enter short when daily RSI > 70 and 6h price closes below 6h VWAP with volume > 1.5x average.
Exit when daily RSI returns to neutral (40-60 range) or opposite extreme is reached.
Designed to capture reversals in both bull and bear markets, avoiding trend-following whipsaws.
Target: 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_RSI_Reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY RSI CALCULATION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 6H VWAP CALCULATION ===
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / (vwap_den + 1e-10)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vwap[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        long_signal = (rsi_1d_aligned[i] < 30 and 
                      close[i] > vwap[i] and 
                      vol_ratio[i] > 1.5)
        
        short_signal = (rsi_1d_aligned[i] > 70 and 
                       close[i] < vwap[i] and 
                       vol_ratio[i] > 1.5)
        
        # Exit conditions: RSI returns to neutral or opposite extreme
        exit_long = (position == 1 and 
                    (rsi_1d_aligned[i] > 40 or rsi_1d_aligned[i] > 70))
        exit_short = (position == -1 and 
                     (rsi_1d_aligned[i] < 60 or rsi_1d_aligned[i] < 30))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals