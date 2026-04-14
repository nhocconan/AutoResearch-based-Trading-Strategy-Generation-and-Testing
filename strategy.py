#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day EMA trend with 4h RSI mean reversion and volume spike.
# Long when price < 1-day EMA200 AND 4h RSI(14) < 30 AND 4h volume > 2x 20-period average.
# Short when price > 1-day EMA200 AND 4h RSI(14) > 70 AND 4h volume > 2x 20-period average.
# Exit when price crosses the 1-day EMA200 OR RSI returns to neutral (40-60).
# Uses 1-day EMA200 for long-term trend bias and 4h RSI/volume for overextended bounces in both bull and bear markets.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag while capturing mean reversion within trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(200)
    ema_200 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        # Simple average for first value
        ema_200[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200[i] = (close_1d[i] * 2 + ema_200[i-1] * 199) / 201
    
    # Load 4h data ONCE for RSI and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h RSI(14)
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    # Wilder's smoothing for RSI
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[1:15])  # Skip first element
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # Handle no loss case
    
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(19, len(volume_4h)):
        vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
    
    # Align indicators to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(200, 30)  # Need 1d and 4h data
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        volume_ratio = volume_4h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for mean reversion entries with volume spike
            # Long: price below EMA200 AND RSI oversold AND volume spike
            if (close[i] < ema_200_aligned[i] and 
                rsi_aligned[i] < 30 and 
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: price above EMA200 AND RSI overbought AND volume spike
            elif (close[i] > ema_200_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses EMA200 or RSI returns to neutral
            if (close[i] > ema_200_aligned[i] or 
                (rsi_aligned[i] >= 40 and rsi_aligned[i] <= 60)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses EMA200 or RSI returns to neutral
            if (close[i] < ema_200_aligned[i] or 
                (rsi_aligned[i] >= 40 and rsi_aligned[i] <= 60)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_EMA200_RSI_Volume_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0