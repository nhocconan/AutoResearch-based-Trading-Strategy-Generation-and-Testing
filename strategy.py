#!/usr/bin/env python3
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
    
    # === 1d Price Range (14-period high-low) for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range (high - low) and its 14-period average
    daily_range = high_1d - low_1d
    range_avg_14 = np.full_like(daily_range, np.nan)
    for i in range(len(daily_range)):
        if i >= 13:
            range_avg_14[i] = np.mean(daily_range[i-13:i+1])
        elif i > 0:
            range_avg_14[i] = np.mean(daily_range[max(0, i-6):i+1])
        else:
            range_avg_14[i] = daily_range[0]
    
    # === 1d RSI (14-period) for momentum ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[avg_loss == 0] = 100
    
    # Align indicators to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    range_avg_14_aligned = align_htf_to_ltf(prices, df_1d, range_avg_14)
    
    # === 4h Volume confirmation ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period average volume on 4h timeframe
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_confirm = volume_4h > vol_ma_20 * 1.5
    vol_confirm_aligned = align_htf_to_ltf(prices, df_4h, vol_confirm)
    
    # === Session filter: 08:00-20:00 UTC ===
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(range_avg_14_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume confirmation AND in session
        if position == 0:
            # Long: RSI < 35 (oversold) + volatility filter (range > 0) + volume confirmation
            if (rsi_1d_aligned[i] < 35 and 
                range_avg_14_aligned[i] > 0 and  # volatility filter
                vol_confirm_aligned[i]):
                signals[i] = 0.20
                position = 1
                continue
            # Short: RSI > 65 (overbought) + volatility filter + volume confirmation
            elif (rsi_1d_aligned[i] > 65 and 
                  range_avg_14_aligned[i] > 0 and  # volatility filter
                  vol_confirm_aligned[i]):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses above 65 (overbought)
            if rsi_1d_aligned[i] > 65:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI crosses below 35 (oversold)
            if rsi_1d_aligned[i] < 35:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_Volume_Confirm_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0