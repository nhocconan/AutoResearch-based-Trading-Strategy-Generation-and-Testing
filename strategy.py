#!/usr/bin/env python3
# Hypothesis: 4h KAMA trend + RSI(2) mean reversion + volume confirmation
# Long when KAMA rising, RSI(2) < 10, and volume > 1.5x average
# Short when KAMA falling, RSI(2) > 90, and volume > 1.5x average
# Exit when RSI(2) crosses 50 (mean reversion complete)
# Uses adaptive trend (KAMA) for direction, extreme short-term RSI for entry, volume for conviction
# Designed to capture mean reversion within trending markets with controlled frequency
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_KAMA_RSI2_MeanReversion_Volume"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 4h close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[:20])  # placeholder, will compute properly below
    
    # Proper KAMA calculation
    close_series = pd.Series(close)
    change = close_series.diff().abs()
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(2) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=2, min_periods=2).mean()
    avg_loss = loss.rolling(window=2, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for RSI and volatility calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising, RSI(2) < 10 (oversold), volume spike
            if (kama[i] > kama[i-1] and 
                rsi_values[i] < 10 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI(2) > 90 (overbought), volume spike
            elif (kama[i] < kama[i-1] and 
                  rsi_values[i] > 90 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI(2) crosses above 50 (mean reversion complete)
            if rsi_values[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI(2) crosses below 50 (mean reversion complete)
            if rsi_values[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals