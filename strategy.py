# 4H_RSI_SMA200_CONFIRMATION_V1
# Hypothesis: Enter long when RSI crosses above 50 and price > 200-period SMA, short when RSI crosses below 50 and price < 200-period SMA.
# Uses 1d SMA200 for trend direction to avoid whipsaw in ranging markets. RSI provides timely entries.
# Designed for low trade frequency (~20-40/year) to minimize fee drag. Works in both bull and bear markets.
timeframe = "4h"
name = "4H_RSI_SMA200_CONFIRMATION_V1"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d SMA200 for trend direction
    df_1d = get_htf_data(prices, '1d')
    sma200_1d = pd.Series(df_1d['close']).rolling(window=200, min_periods=200).mean().values
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        if np.isnan(rsi[i]) or np.isnan(sma200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI crosses above 50 and price > 1d SMA200
            if rsi[i] > 50 and rsi[i-1] <= 50 and close[i] > sma200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 50 and price < 1d SMA200
            elif rsi[i] < 50 and rsi[i-1] >= 50 and close[i] < sma200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses below 50
            if rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses above 50
            if rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals