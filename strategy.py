#!/usr/bin/env python3
# 4h_12h_RSI_MeanReversion_with_VolumeFilter
# Hypothesis: RSI mean reversion on 4h with 12h trend filter and volume spike confirmation.
# In bull markets, buy oversold dips in uptrend; in bear markets, sell overbought rallies in downtrend.
# Volume filter avoids false signals during low-volume chop. Target: 20-40 trades/year.

name = "4h_12h_RSI_MeanReversion_with_VolumeFilter"
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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h RSI (14-period)
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_12h = 100 - (100 / (1 + rs))
    # For first 13 values, RSI is undefined; set to 50 (neutral)
    rsi_12h[:13] = 50
    
    # Calculate 12h EMA20 for trend filter
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume average for spike detection (24 periods = 4 days)
    vol_ma_12h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align 12h indicators to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(ema20_12h_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.3 * 12h average volume
        volume_spike = volume[i] > 1.3 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND price above 12h EMA20 (uptrend) with volume spike
            if rsi_12h_aligned[i] < 30 and close[i] > ema20_12h_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) AND price below 12h EMA20 (downtrend) with volume spike
            elif rsi_12h_aligned[i] > 70 and close[i] < ema20_12h_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI > 50 (mean reversion complete) or trend changes
            if rsi_12h_aligned[i] > 50 or close[i] < ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI < 50 (mean reversion complete) or trend changes
            if rsi_12h_aligned[i] < 50 or close[i] > ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals