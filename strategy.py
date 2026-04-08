#!/usr/bin/env python3
"""
6h_12h_1d_pullback_volume_v1
Hypothesis: Use 12h EMA for trend direction and 1d ATR for volatility filter, enter on 6h pullbacks to EMA with volume confirmation.
Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_pullback_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA(21) for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.maximum(np.abs(low_1d - np.roll(close_1d, 1)), tr1)
    tr2[0] = high_1d[0] - low_1d[0]  # First TR
    atr_1d = pd.Series(tr2).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 12h EMA or volatility too low
            if close[i] < ema_12h_aligned[i] or atr_1d_aligned[i] < np.mean(atr_1d_aligned[max(0, i-20):i+1]) * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above 12h EMA or volatility too low
            if close[i] > ema_12h_aligned[i] or atr_1d_aligned[i] < np.mean(atr_1d_aligned[max(0, i-20):i+1]) * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price pulls back to 12h EMA from above with volume and volatility sufficient
            if (close[i] > ema_12h_aligned[i] and 
                low[i] <= ema_12h_aligned[i] * 1.005 and  # Allow small penetration
                close[i] >= ema_12h_aligned[i] * 0.995 and
                vol_confirm[i] and 
                atr_1d_aligned[i] > np.mean(atr_1d_aligned[max(0, i-20):i+1]) * 0.8):
                position = 1
                signals[i] = 0.25
            # Short entry: price pulls back to 12h EMA from below with volume and volatility sufficient
            elif (close[i] < ema_12h_aligned[i] and 
                  high[i] >= ema_12h_aligned[i] * 0.995 and  # Allow small penetration
                  close[i] <= ema_12h_aligned[i] * 1.005 and
                  vol_confirm[i] and 
                  atr_1d_aligned[i] > np.mean(atr_1d_aligned[max(0, i-20):i+1]) * 0.8):
                position = -1
                signals[i] = -0.25
    
    return signals