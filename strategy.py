#!/usr/bin/env python3
"""
12h_1d_momentum_volume_reversal_v1
Hypothesis: On 12h timeframe, combine momentum (12-period RSI) with volume confirmation and 1d trend filter.
Go long when RSI < 30 (oversold) + volume spike + price above 1d EMA50.
Go short when RSI > 70 (overbought) + volume spike + price below 1d EMA50.
Exit when RSI returns to neutral zone (40-60).
Designed to capture mean reversals in ranging markets while avoiding trend-following whipsaws.
Target: 15-35 trades/year per symbol (60-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_momentum_volume_reversal_v1"
timeframe = "12h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[12] = np.mean(gain[1:13])
    avg_loss[12] = np.mean(loss[1:13])
    
    for i in range(13, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 11 + gain[i]) / 12
        avg_loss[i] = (avg_loss[i-1] * 11 + loss[i]) / 12
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral zone (40-60)
            if rsi[i] >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral zone (40-60)
            if rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold) + volume spike + price above 1d EMA50
            if rsi[i] < 30 and vol_confirm[i] and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70 (overbought) + volume spike + price below 1d EMA50
            elif rsi[i] > 70 and vol_confirm[i] and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals