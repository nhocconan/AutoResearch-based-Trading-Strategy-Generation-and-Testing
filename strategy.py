#!/usr/bin/env python3
"""
1d 200-day EMA Trend + RSI Mean Reversion + Volume Spike
Hypothesis: In trending markets (price > 200EMA), pullbacks to RSI oversold/overbought levels with volume confirmation offer high-probability reversals. Works in both bull (buy dips) and bear (sell rallies) by following the 200EMA trend. Low frequency due to strict 200EMA filter and RSI extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (more robust than daily)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 200-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_200 = np.zeros_like(close_1w)
    ema_200[0] = close_1w[0]
    alpha = 2.0 / (200 + 1)
    for i in range(1, len(close_1w)):
        ema_200[i] = ema_200[i-1] + alpha * (close_1w[i] - ema_200[i-1])
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Calculate RSI(14) on daily close
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        
        # First average
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        # Wilder smoothing
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        ema_val = ema_200_aligned[i]
        rsi_val = rsi[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: price above weekly 200EMA (uptrend) + RSI oversold + volume spike
            if (close[i] > ema_val and 
                rsi_val < 30 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly 200EMA (downtrend) + RSI overbought + volume spike
            elif (close[i] < ema_val and 
                  rsi_val > 70 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or price crosses below weekly 200EMA
            if rsi_val > 70 or close[i] < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or price crosses above weekly 200EMA
            if rsi_val < 30 or close[i] > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_200EMA_RSI_MeanReversion_VolumeSpike"
timeframe = "1d"
leverage = 1.0