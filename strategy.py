#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d RSI and volume confirmation
# RSI(14) < 30 indicates oversold conditions, RSI > 70 overbought
# Combined with volume spike (current volume > 1.5x 20-period average) for confirmation
# Works in both bull and bear markets by capturing mean reversion during extremes
# Uses 1d RSI for signal and volume for confirmation - avoids overtrading

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI (14)
    rsi_length = 14
    rsi_src = df_1d['close'].values
    delta = np.diff(rsi_src, prepend=rsi_src[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_length, adjust=False, min_periods=rsi_length).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_length, adjust=False, min_periods=rsi_length).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate volume moving average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need enough for RSI and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: RSI oversold + volume spike
            if rsi_aligned[i] < 30 and volume_spike:
                position = 1
                signals[i] = position_size
            # Enter short: RSI overbought + volume spike
            elif rsi_aligned[i] > 70 and volume_spike:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or RSI > 50
            if rsi_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or RSI < 50
            if rsi_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dRSI_Volume_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0