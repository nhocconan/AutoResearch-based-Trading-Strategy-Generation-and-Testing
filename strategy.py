#!/usr/bin/env python3
# Hypothesis: 4h timeframe with 1-day RSI mean reversion and volume confirmation.
# In overbought/oversold conditions (RSI > 70 or < 30), price tends to revert to the mean.
# Enters long when RSI < 30 and price closes above the 20-period EMA, short when RSI > 70 and price closes below the 20-period EMA.
# Uses volume confirmation: current volume must be above the 20-period average volume.
# Exits when RSI returns to neutral territory (40-60) or price crosses the 20-period EMA in the opposite direction.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "4h_RSI_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day RSI (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Calculate 20-period EMA for trend filter
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or
            np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: oversold (RSI < 30) + price above EMA20 + volume above average
            if rsi_1d_aligned[i] < 30 and close[i] > ema_20[i] and volume[i] > volume_ma[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought (RSI > 70) + price below EMA20 + volume above average
            elif rsi_1d_aligned[i] > 70 and close[i] < ema_20[i] and volume[i] > volume_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or price crosses below EMA20
            if rsi_1d_aligned[i] >= 40 or close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or price crosses above EMA20
            if rsi_1d_aligned[i] <= 60 or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals