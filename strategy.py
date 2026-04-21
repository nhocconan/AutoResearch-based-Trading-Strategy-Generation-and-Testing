#!/usr/bin/env python3
"""
6h_EMA34_RSI2_VolumeSpike
Hypothesis: Mean reversion on 6h timeframe using RSI(2) extreme readings filtered by 1d EMA34 trend and volume spikes.
Works in bull/bear: In uptrend (price>EMA34), buy RSI<10 pullbacks; in downtrend (price<EMA34), sell RSI>90 rallies.
Volume spike confirms institutional interest. Target: 12-30 trades/year per symbol (50-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate RSI(2) on 6h close
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2  # 2-period Wilder's
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[0] = 50  # neutral for first bar
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    volume = prices['volume'].values
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20:i])
    
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if indicators not ready
        if np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price > 1d EMA34 (uptrend) AND RSI < 10 (extreme oversold) AND volume spike
            if (price > ema_34_1d_aligned[i] and 
                rsi_val < 10 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price < 1d EMA34 (downtrend) AND RSI > 90 (extreme overbought) AND volume spike
            elif (price < ema_34_1d_aligned[i] and 
                  rsi_val > 90 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) or price < 1d EMA34 (trend change)
            if rsi_val > 50 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) or price > 1d EMA34 (trend change)
            if rsi_val < 50 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA34_RSI2_VolumeSpike"
timeframe = "6h"
leverage = 1.0