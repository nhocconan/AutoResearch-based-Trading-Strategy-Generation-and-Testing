#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 12h RSI(14) extreme + 1w EMA50 trend filter + volume confirmation.
# Long when RSI < 30 (oversold) and price > 1w EMA50 and volume > 1.5x average.
# Short when RSI > 70 (overbought) and price < 1w EMA50 and volume > 1.5x average.
# Exit when RSI crosses back to neutral (40 for long exit, 60 for short exit).
# Uses 1w EMA50 for trend filter to avoid counter-trend trades in strong trends.
# Targets 15-25 trades per year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate RSI(14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    # First average
    if len(gain) > rsi_period:
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    # Wilder's smoothing
    for i in range(rsi_period + 1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.zeros_like(avg_gain)
    rsi = np.zeros_like(avg_gain)
    for i in range(rsi_period, len(avg_gain)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Align 1w EMA50 to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI, EMA50, and volume MA20
    start_idx = max(rsi_period, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_now = rsi[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price > 1w EMA50 and volume filter
            if (rsi_now < 30 and 
                price > ema_1w_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: RSI > 70 (overbought) and price < 1w EMA50 and volume filter
            elif (rsi_now > 70 and 
                  price < ema_1w_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses back above 40
            if rsi_now > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI crosses back below 60
            if rsi_now < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_RSI14_Extreme_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0