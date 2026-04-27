#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h RSI extremes with 1d EMA trend filter and volume confirmation.
# Long when RSI(12h) < 30 (oversold) with 1d EMA50 uptrend and volume > 1.5x average.
# Short when RSI(12h) > 70 (overbought) with 1d EMA50 downtrend and volume > 1.5x average.
# Exit when RSI crosses back to neutral zone (40 for long exit, 60 for short exit).
# Uses 12h RSI for fewer, higher-quality signals and 1d EMA for trend alignment to work in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for RSI calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h RSI (14-period)
    rsi_period = 14
    delta = np.diff(close_12h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_12h), np.nan)
    avg_loss = np.full(len(close_12h), np.nan)
    
    if len(gain) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        for i in range(rsi_period + 1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
    
    rs = np.full(len(close_12h), np.nan)
    rsi_12h = np.full(len(close_12h), np.nan)
    valid = (avg_loss != 0) & ~np.isnan(avg_gain) & ~np.isnan(avg_loss)
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    rsi_12h[valid] = 100 - (100 / (1 + rs[valid]))
    rsi_12h[avg_loss == 0] = 100
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Align 12h RSI and 1d EMA to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI, EMA50, and volume MA20
    start_idx = max(19, 19)  # volume MA20 needs 19, others handled by alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        rsi = rsi_12h_aligned[i]
        ema = ema_1d_aligned[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: RSI < 30 (oversold) with 1d EMA50 uptrend and volume filter
            if (rsi < 30 and 
                price > ema and vol_filter):
                signals[i] = size
                position = 1
            # Short: RSI > 70 (overbought) with 1d EMA50 downtrend and volume filter
            elif (rsi > 70 and 
                  price < ema and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses back above 40 (leaving oversold zone)
            if rsi > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI crosses back below 60 (leaving overbought zone)
            if rsi < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSI12h_Extreme_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0