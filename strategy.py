#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d RSI + Volume Spike + ATR Stop
# Williams Alligator (jaw=13, teeth=8, lips=5) identifies trend direction.
# Long when lips > teeth > jaw and RSI < 30 (oversold bounce) with volume spike.
# Short when lips < teeth < jaw and RSI > 70 (overbought rejection) with volume spike.
# Uses 1d RSI for higher timeframe confirmation to avoid false signals.
# Target: 50-150 total trades over 4 years (12-38/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Williams Alligator on 4h
    # Jaw (13-period SMMA of median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    # Teeth (8-period SMMA of median price)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    # Lips (5-period SMMA of median price)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Volume spike detection (20-period median)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_median[i])):
            continue
        
        # Long entry: Lips > Teeth > Jaw (bullish alignment) + RSI < 30 + Volume spike
        if (lips[i] > teeth[i] and teeth[i] > jaw[i] and
            rsi_1d_aligned[i] < 30 and
            volume[i] > 1.5 * vol_median[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Lips < Teeth < Jaw (bearish alignment) + RSI > 70 + Volume spike
        elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and
              rsi_1d_aligned[i] > 70 and
              volume[i] > 1.5 * vol_median[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Reverse Alligator alignment or RSI returns to neutral zone
        elif position == 1 and (lips[i] < teeth[i] or rsi_1d_aligned[i] > 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (lips[i] > teeth[i] or rsi_1d_aligned[i] < 50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0