#!/usr/bin/env python3
"""
1d RSI-40/60 Pullback to 20-day SMA + Volume Confirmation
Hypothesis: In trending markets, price pulls back to the 20-day SMA before continuing.
RSI 40-60 range identifies pullbacks in established trends (avoiding overbought/oversold extremes).
Volume confirmation ensures institutional participation. Works in bull/bear by following trend.
Low trade frequency: only takes trades in alignment with 200-day trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 200-day SMA for trend filter (using daily data)
    df_200d = get_htf_data(prices, '1d')
    if len(df_200d) < 200:
        return np.zeros(n)
    close_200d = df_200d['close'].values
    sma_200 = np.zeros_like(close_200d)
    for i in range(len(close_200d)):
        if i < 199:
            sma_200[i] = np.nan
        else:
            sma_200[i] = np.mean(close_200d[i-199:i+1])
    sma_200_aligned = align_htf_to_ltf(prices, df_200d, sma_200)
    
    # Get 20-day SMA for pullback target (using daily data)
    df_20d = get_htf_data(prices, '1d')
    if len(df_20d) < 20:
        return np.zeros(n)
    close_20d = df_20d['close'].values
    sma_20 = np.zeros_like(close_20d)
    for i in range(len(close_20d)):
        if i < 19:
            sma_20[i] = np.nan
        else:
            sma_20[i] = np.mean(close_20d[i-19:i+1])
    sma_20_aligned = align_htf_to_ltf(prices, df_20d, sma_20)
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close_200d, prepend=close_200d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    def rsi_wilder(gain, loss, period=14):
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        if len(gain) < period:
            return np.full_like(gain, 50.0)
        # Initial average
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        # Wilder smoothing
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = rsi_wilder(gain, loss, 14)
    rsi_aligned = align_htf_to_ltf(prices, df_200d, rsi)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 19:
            vol_ma_20[i] = np.mean(volume[max(0, i-18):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma_20[i] = np.mean(volume[i-18:i+1])
    vol_confirm = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need 200-day trend
    
    for i in range(start_idx, n):
        if np.isnan(sma_200_aligned[i]) or np.isnan(sma_20_aligned[i]) or np.isnan(rsi_aligned[i]):
            signals[i] = 0.0
            continue
        
        sma200_val = sma_200_aligned[i]
        sma20_val = sma_20_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Enter long: price above 200-day SMA (uptrend), pulling back to 20-day SMA,
            # RSI in 40-60 range (not overbought), with volume confirmation
            if (close[i] > sma200_val and 
                close[i] <= sma20_val * 1.02 and  # Allow small overshoot above 20-day SMA
                close[i] >= sma20_val * 0.98 and  # Allow small undershoot below 20-day SMA
                40 <= rsi_val <= 60 and
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price below 200-day SMA (downtrend), pulling back to 20-day SMA,
            # RSI in 40-60 range (not oversold), with volume confirmation
            elif (close[i] < sma200_val and 
                  close[i] <= sma20_val * 1.02 and
                  close[i] >= sma20_val * 0.98 and
                  40 <= rsi_val <= 60 and
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 20-day SMA or RSI > 70 (overbought)
            if close[i] < sma20_val * 0.98 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 20-day SMA or RSI < 30 (oversold)
            if close[i] > sma20_val * 1.02 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_Pullback_to_SMA20_Volume"
timeframe = "1d"
leverage = 1.0