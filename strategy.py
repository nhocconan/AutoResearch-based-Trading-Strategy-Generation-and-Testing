#!/usr/bin/env python3
"""
Hypothesis: 1d RSI(14) mean reversion with 1w EMA trend filter.
- RSI < 30 for long, RSI > 70 for short captures mean reversion in both bull/bear markets
- 1w EMA(50) filters for trend direction, avoids counter-trend trades
- Volume spike (volume > 1.5x 20-period avg) confirms participation
- Exit on opposite RSI level or trend change
- Target: 10-20 trades/year to avoid fee drag
- Uses discrete position sizing (0.25) to minimize churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on weekly data
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 + ema_1w[i-1] * 49) / 51
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate RSI(14) on daily data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Wilder's smoothing for RSI
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(30, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: RSI < 30 + price above weekly EMA + volume spike
            if (rsi[i] < 30 and 
                close[i] > ema_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: RSI > 70 + price below weekly EMA + volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 OR price below weekly EMA (trend change)
            if (rsi[i] > 70 or 
                close[i] < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 30 OR price above weekly EMA (trend change)
            if (rsi[i] < 30 or 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI14_MeanRev_EMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0