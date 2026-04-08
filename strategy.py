#!/usr/bin/env python3
"""
1h RSI Pullback with 4h Trend Filter and Volume Confirmation
Hypothesis: In strong trends (4h), pullbacks to RSI(14) < 30 (long) or > 70 (short) offer high-probability entries.
Volume > 1.5x average confirms conviction. Works in bull/bear by following 4h trend direction.
Target: 15-30 trades/year (~60-120 total over 4 years) with 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = df_4h['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend reversal
            if rsi[i] >= 70 or close[i] < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend reversal
            if rsi[i] <= 30 or close[i] > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long pullback in uptrend: RSI < 30 + volume
            if (rsi[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short pullback in downtrend: RSI > 70 + volume
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals