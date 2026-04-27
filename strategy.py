#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and session filter
# Uses RSI(14) on 1h for overbought/oversold signals, 4h EMA(50) for trend direction,
# and restricts trading to 08:00-20:00 UTC to avoid low-volume periods.
# In uptrend (price > 4h EMA50), look for RSI < 30 for long entries.
# In downtrend (price < 4h EMA50), look for RSI > 70 for short entries.
# Exits when RSI returns to neutral (40-60 range) or opposite extreme.
# Designed for 1h timeframe with low trade frequency (15-30/year) to minimize fee drag.
# Works in both bull and bear markets by following the 4h trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h close
    close_4h = df_4h['close'].values
    ema_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        multiplier = 2 / (50 + 1)
        ema_4h[0] = close_4h[0]
        for i in range(1, len(close_4h)):
            ema_4h[i] = (close_4h[i] * multiplier) + (ema_4h[i-1] * (1 - multiplier))
    
    # Align 4h EMA to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate RSI(14) on 1h
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    if len(gain) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[1:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[1:rsi_period])
        for i in range(rsi_period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Start after warmup period
    start_idx = max(50, rsi_period)
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            continue
            
        price = close[i]
        rsi_val = rsi[i]
        ema_val = ema_4h_aligned[i]
        
        if position == 0:
            # Look for entry: RSI extreme in direction of 4h trend
            if price > ema_val and rsi_val < 30:  # Uptrend + oversold -> long
                signals[i] = size
                position = 1
            elif price < ema_val and rsi_val > 70:  # Downtrend + overbought -> short
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI returns to neutral or becomes overbought
            if rsi_val >= 40 and rsi_val <= 60:  # Return to neutral
                signals[i] = 0.0
                position = 0
            elif rsi_val > 70:  # Overbought - take profit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI returns to neutral or becomes oversold
            if rsi_val >= 40 and rsi_val <= 60:  # Return to neutral
                signals[i] = 0.0
                position = 0
            elif rsi_val < 30:  # Oversold - take profit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI_MeanReversion_4hEMATrend_Session"
timeframe = "1h"
leverage = 1.0