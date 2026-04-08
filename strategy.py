#!/usr/bin/env python3
"""
1h_4h_1d_trend_pullback_volume_v1
Hypothesis: Use 4h EMA for trend direction and 1d ATR for volatility filter, enter on 1h pullbacks to EMA with volume confirmation.
Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
Target: 15-35 trades/year per symbol (60-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_trend_pullback_volume_v1"
timeframe = "1h"
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
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(21) for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.maximum(np.abs(low_1d - np.roll(close_1d, 1)), tr1)
    tr2[0] = high_1d[0] - low_1d[0]  # First TR
    atr_1d = pd.Series(tr2).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: volume > 1.3x average of last 24 periods
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 4h EMA or volatility too low
            if close[i] < ema_4h_aligned[i] or atr_1d_aligned[i] < np.mean(atr_1d_aligned[max(0, i-24):i+1]) * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above 4h EMA or volatility too low
            if close[i] > ema_4h_aligned[i] or atr_1d_aligned[i] < np.mean(atr_1d_aligned[max(0, i-24):i+1]) * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price pulls back to 4h EMA from above with volume and volatility sufficient
            if (close[i] > ema_4h_aligned[i] and 
                low[i] <= ema_4h_aligned[i] * 1.002 and  # Allow small penetration
                close[i] >= ema_4h_aligned[i] * 0.998 and
                vol_confirm[i] and 
                atr_1d_aligned[i] > np.mean(atr_1d_aligned[max(0, i-24):i+1]) * 0.8):
                position = 1
                signals[i] = 0.20
            # Short entry: price pulls back to 4h EMA from below with volume and volatility sufficient
            elif (close[i] < ema_4h_aligned[i] and 
                  high[i] >= ema_4h_aligned[i] * 0.998 and  # Allow small penetration
                  close[i] <= ema_4h_aligned[i] * 1.002 and
                  vol_confirm[i] and 
                  atr_1d_aligned[i] > np.mean(atr_1d_aligned[max(0, i-24):i+1]) * 0.8):
                position = -1
                signals[i] = -0.20
    
    return signals