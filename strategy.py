#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Daily RSI with Volume and Momentum Filter
# Hypothesis: Daily RSI extremes (>70 or <30) combined with volume spike and
# 1h momentum (price > 20 EMA) provides high-probability reversals in both
# bull and bear markets. In bull markets: buy RSI<30 with momentum. In bear
# markets: sell RSI>70 with momentum. Uses daily timeframe for signal direction
# and 1h for entry timing to reduce noise and control trade frequency.
# Target: 15-30 trades/year (60-120 over 4 years).

name = "1h_daily_rsi_volume_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    daily_close = df_daily['close'].values
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    rsi = np.roll(rsi, 1)
    if len(rsi) > 1:
        rsi[0] = 50  # neutral
    
    # Align to 1h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    
    # Momentum filter: price > 20 EMA on 1h
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Volume filter: volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: RSI > 60 or loss of momentum
            if rsi_aligned[i] > 60 or close[i] <= ema_20[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit conditions: RSI < 40 or loss of momentum
            if rsi_aligned[i] < 40 or close[i] >= ema_20[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Long entry: RSI < 30 with momentum and volume
            if (rsi_aligned[i] < 30 and close[i] > ema_20[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI > 70 with momentum and volume
            elif (rsi_aligned[i] > 70 and close[i] < ema_20[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals