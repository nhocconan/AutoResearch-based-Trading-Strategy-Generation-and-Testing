#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day RSI(14) mean reversion with 1-week EMA(34) trend filter and volume confirmation.
# Long when RSI(14) crosses above 30 from below (oversold bounce) with 1-week EMA34 uptrend and volume > 1.5x average.
# Short when RSI(14) crosses below 70 from above (overbought rejection) with 1-week EMA34 downtrend and volume > 1.5x average.
# Exit when RSI(14) crosses back through 50 (mean reversion).
# Uses RSI for precise mean-reversion timing on 1d timeframe, targeting 12-30 trades per year.
# Designed to work in both bull and bear markets via trend filter and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1-week EMA34 for trend filter
    ema_period = 34
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate RSI (14-period)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Initialize first average
    if n >= rsi_period:
        avg_gain[rsi_period - 1] = np.mean(gain[1:rsi_period])
        avg_loss[rsi_period - 1] = np.mean(loss[1:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period, n):
            avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    for i in range(rsi_period - 1, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100.0  # Avoid division by zero
    
    # RSI previous value for crossover detection
    rsi_prev = np.full(n, np.nan)
    rsi_prev[1:] = rsi[:-1]
    
    # Align 1-week EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI, EMA34, and volume MA20
    start_idx = max(rsi_period, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(rsi_prev[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: RSI crosses above 30 from below with 1-week EMA34 uptrend and volume filter
            if (rsi_prev[i] <= 30 and rsi[i] > 30 and 
                price > ema_1w_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: RSI crosses below 70 from above with 1-week EMA34 downtrend and volume filter
            elif (rsi_prev[i] >= 70 and rsi[i] < 70 and 
                  price < ema_1w_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses below 50 from above
            if rsi_prev[i] >= 50 and rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI crosses above 50 from below
            if rsi_prev[i] <= 50 and rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RSI14_MeanReversion_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0