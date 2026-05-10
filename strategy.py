#!/usr/bin/env python3
"""
4H_RSI_Divergence_With_Volume_Confirmation
Hypothesis: Uses RSI divergence (price vs RSI) combined with volume confirmation to capture trend reversals.
Works in both bull and bear markets by detecting exhaustion points. Uses 4h timeframe with 1h RSI for
divergence detection and volume spike confirmation. Target: 20-40 trades/year to minimize fee drag.
"""

name = "4H_RSI_Divergence_With_Volume_Confirmation"
timeframe = "4h"
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
    
    # Get 1h data for RSI calculation (more responsive than 4h RSI)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1h closes
    delta = pd.Series(df_1h['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_values)
    
    # Volume filter: volume > 1.8x 20-period average on 4h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for RSI and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1h_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for RSI divergence (need at least 3 bars back)
        if i >= 3:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bull_div = (low[i] < low[i-2] and 
                       rsi_1h_aligned[i] > rsi_1h_aligned[i-2] and
                       rsi_1h_aligned[i] < 40)  # RSI not overbought
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bear_div = (high[i] > high[i-2] and 
                       rsi_1h_aligned[i] < rsi_1h_aligned[i-2] and
                       rsi_1h_aligned[i] > 60)  # RSI not oversold
            
            if position == 0:
                # Long entry: bullish divergence + volume spike
                if bull_div and volume[i] > vol_threshold[i]:
                    signals[i] = 0.25
                    position = 1
                # Short entry: bearish divergence + volume spike
                elif bear_div and volume[i] > vol_threshold[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Long exit: bearish divergence or RSI overbought
                if bear_div or rsi_1h_aligned[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: bullish divergence or RSI oversold
                if bull_div or rsi_1h_aligned[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals