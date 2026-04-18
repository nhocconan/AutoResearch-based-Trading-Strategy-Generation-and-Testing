#!/usr/bin/env python3
"""
1h_RSI_Divergence_4hTrend
Hypothesis: Uses 4h EMA50 for trend direction and 1h RSI divergence for entry timing.
Trades only during high-liquidity session (08-20 UTC) to reduce noise.
Targets 15-30 trades/year by requiring trend alignment + divergence + volume confirmation.
Works in bull/bear: trend filter avoids counter-trend trades, divergence catches reversals within trend.
"""

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema_50_4h = np.zeros_like(close_4h)
    ema_50_4h[:] = np.nan
    if len(close_4h) >= 50:
        k = 2 / (50 + 1)
        ema_50_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = close_4h[i] * k + ema_50_4h[i-1] * (1 - k)
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(gain)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * i + gain[i]) / (i + 1)
                avg_loss[i] = (avg_loss[i-1] * i + loss[i]) / (i + 1)
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 50  # Warmup for EMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bull_div = (low[i] < low[i-1] and low[i] < low[i-2] and 
                       rsi[i] > rsi[i-1] and rsi[i] > rsi[i-2])
            # Bearish divergence: price makes higher high, RSI makes lower high
            bear_div = (high[i] > high[i-1] and high[i] > high[i-2] and 
                       rsi[i] < rsi[i-1] and rsi[i] < rsi[i-2])
            
            # Long: bullish divergence in uptrend (price above 4h EMA50)
            if bull_div and close[i] > ema_50_4h_aligned[i] and vol_spike[i]:
                signals[i] = 0.20
                position = 1
                bars_since_entry = 0
            # Short: bearish divergence in downtrend (price below 4h EMA50)
            elif bear_div and close[i] < ema_50_4h_aligned[i] and vol_spike[i]:
                signals[i] = -0.20
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: opposite divergence or trend change
            bear_div = (high[i] > high[i-1] and high[i] > high[i-2] and 
                       rsi[i] < rsi[i-1] and rsi[i] < rsi[i-2])
            if bear_div or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: opposite divergence or trend change
            bull_div = (low[i] < low[i-1] and low[i] < low[i-2] and 
                       rsi[i] > rsi[i-1] and rsi[i] > rsi[i-2])
            if bull_div or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_Divergence_4hTrend"
timeframe = "1h"
leverage = 1.0