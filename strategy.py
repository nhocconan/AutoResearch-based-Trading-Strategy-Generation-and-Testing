#!/usr/bin/env python3
# 4h_rsi_momentum_volume_v1
# Hypothesis: 4-hour RSI momentum with volume confirmation works in both bull and bear markets.
# Uses RSI(14) > 60 for long momentum and < 40 for short momentum, requiring volume > 1.5x 20-period average.
# Includes dynamic exit: reverse signal or volume drop below average. Position size 0.25.
# Target: 20-40 trades/year (80-160 over 4 years) with balanced win/loss.

import numpy as np
import pandas as pd

name = "4h_rsi_momentum_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI < 40 (momentum lost) or volume drops below average
            if rsi[i] < 40 or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI > 60 (momentum lost) or volume drops below average
            if rsi[i] > 60 or volume[i] < vol_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI > 60 with volume confirmation (bullish momentum)
            if rsi[i] > 60 and volume[i] > vol_threshold[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: RSI < 40 with volume confirmation (bearish momentum)
            elif rsi[i] < 40 and volume[i] > vol_threshold[i]:
                position = -1
                signals[i] = -0.25
    
    return signals