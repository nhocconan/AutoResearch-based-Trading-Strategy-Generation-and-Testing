#!/usr/bin/env python3
"""
1d_RSI_Trend_Filter_With_Volume
Hypothesis: On daily timeframe, RSI(14) below 40 with price above 200-day EMA and volume confirmation generates long signals; RSI above 60 with price below 200-day EMA and volume confirmation generates short signals. Trend filter avoids counter-trend trades, reducing whipsaw in sideways markets. Designed for low trade frequency to minimize fee drag, targeting 10-20 trades/year.
"""

name = "1d_RSI_Trend_Filter_With_Volume"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 200-day EMA trend filter (using 1d data aligned to 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: current volume > 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 40, price above 200 EMA, volume confirmation
            if rsi[i] < 40 and close[i] > ema_200_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 60, price below 200 EMA, volume confirmation
            elif rsi[i] > 60 and close[i] < ema_200_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses above 60 (overbought) or price below 200 EMA
            if rsi[i] > 60 or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses below 40 (oversold) or price above 200 EMA
            if rsi[i] < 40 or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals