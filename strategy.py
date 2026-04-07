#!/usr/bin/env python3
"""
4h_rsi_pullback_1d_trend_volume_v1
Hypothesis: On 4h timeframe, RSI pullbacks in the direction of the daily EMA20 trend with volume confirmation provide high-probability entries. 
In bull markets: buy when RSI < 30 and price > daily EMA20 with volume spike.
In bear markets: sell when RSI > 70 and price < daily EMA20 with volume spike.
Uses daily trend filter to avoid counter-trend trades, and volume to confirm momentum.
Target: 20-50 trades per year on 4h with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA20 for trend filter
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # RSI(14) on 4h closes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Trend filter
        above_ema20 = close[i] > ema20_1d_aligned[i]
        below_ema20 = close[i] < ema20_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (momentum fade) or trend turns bearish with volume
            if rsi_values[i] > 50 or (below_ema20 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (momentum fade) or trend turns bullish with volume
            if rsi_values[i] < 50 or (above_ema20 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long setup: RSI oversold + uptrend + volume
            if rsi_values[i] < 30 and above_ema20 and vol_spike:
                position = 1
                signals[i] = 0.25
            # Short setup: RSI overbought + downtrend + volume
            elif rsi_values[i] > 70 and below_ema20 and vol_spike:
                position = -1
                signals[i] = -0.25
    
    return signals