# USING SINGLE TIMEFRAME 4H - NO MULTI-TIMEFRAME

#!/usr/bin/env python3
"""
4h_RangeBound_MeanReversion
Hypothesis: In ranging markets (2025-2026 mean-reversion regime), price oscillates between Bollinger Bands.
Enter long at lower band with RSI<30, short at upper band with RSI>70. Exit at opposite band or when RSI reaches 50.
Volume filter avoids false breakouts. Works in both bull (buy dips) and bear (sell rallies) by fading extremes.
Target: 20-30 trades/year via strict entry conditions (BB touch + RSI extreme + volume filter).
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.3 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need BB and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price at lower BB, RSI oversold, volume confirmation
            if (close[i] <= bb_lower[i] and rsi[i] < 30 and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price at upper BB, RSI overbought, volume confirmation
            elif (close[i] >= bb_upper[i] and rsi[i] > 70 and vol_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price reaches middle BB or RSI reaches 50
            if (close[i] >= bb_mid[i] or rsi[i] >= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches middle BB or RSI reaches 50
            if (close[i] <= bb_mid[i] or rsi[i] <= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RangeBound_MeanReversion"
timeframe = "4h"
leverage = 1.0