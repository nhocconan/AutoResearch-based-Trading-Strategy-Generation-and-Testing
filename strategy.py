#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(2) mean reversion with 200-day moving average filter
# In bull markets: buy RSI(2) < 10 when price > 200-day MA
# In bear markets: sell RSI(2) > 90 when price < 200-day MA
# Uses 1-day timeframe for MA filter to avoid look-ahead
# Target: 100-200 total trades over 4 years (25-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    # Load daily data ONCE for 200-day MA filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200-day MA
    ma_200 = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    
    # Align daily MA to 4h timeframe
    ma_200_aligned = align_htf_to_ltf(prices, df_1d, ma_200)
    
    # Calculate 4h RSI(2)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    alpha = 1.0 / 2
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if MA not ready
        if np.isnan(ma_200_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ma = ma_200_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: RSI(2) < 10 and price > 200-day MA
            if rsi_val < 10 and price > ma:
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 and price < 200-day MA
            elif rsi_val > 90 and price < ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI(2) > 50 or price < 200-day MA
            if rsi_val > 50 or price < ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI(2) < 50 or price > 200-day MA
            if rsi_val < 50 or price > ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI2_MA200_MeanReversion"
timeframe = "4h"
leverage = 1.0