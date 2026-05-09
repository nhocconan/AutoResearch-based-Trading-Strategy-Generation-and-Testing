#!/usr/bin/env python3
# Hypothesis: 1d timeframe with 1-week RSI divergence (momentum reversal) and volume confirmation.
# In overbought/oversold conditions (weekly RSI > 70 or < 30), price tends to reverse.
# Enters long when weekly RSI < 30 and price closes above 1-day VWAP with volume > 1.5x average.
# Enters short when weekly RSI > 70 and price closes below 1-day VWAP with volume > 1.5x average.
# Exits when weekly RSI returns to neutral zone (40-60) or price crosses VWAP in opposite direction.
# Target: 20-50 total trades over 4 years (5-12/year) with size 0.25.

name = "1d_WeeklyRSI_Divergence_VWAP_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-week RSI (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    delta = close_1w.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_values = rsi_1w.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Calculate 1-day VWAP
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    
    # Average volume for volume spike filter
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.5 * avg_volume.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for VWAP and volume average
    
    for i in range(start_idx, n):
        # Skip if RSI data not ready
        if np.isnan(rsi_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: oversold weekly RSI + price above VWAP + volume spike
            if (rsi_1w_aligned[i] < 30) and (close[i] > vwap[i]) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought weekly RSI + price below VWAP + volume spike
            elif (rsi_1w_aligned[i] > 70) and (close[i] < vwap[i]) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral OR price crosses below VWAP
            if (rsi_1w_aligned[i] >= 40) or (close[i] < vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral OR price crosses above VWAP
            if (rsi_1w_aligned[i] <= 60) or (close[i] > vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals