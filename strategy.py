#!/usr/bin/env python3
# 1h_4h1d_rsi_pullback_v1
# Hypothesis: 1-hour RSI pullback strategy with 4-hour trend filter and 1-day volume confirmation.
# Long: RSI(14) < 30 (oversold pullback) AND 4h EMA(21) up-trend AND 1-day volume > 1.5x 20-day average volume.
# Short: RSI(14) > 70 (overbought bounce) AND 4h EMA(21) down-trend AND 1-day volume > 1.5x 20-day average volume.
# Exit: RSI crosses back above 50 (long) or below 50 (short).
# Designed to capture mean-reversion within the trend during both bull and bear markets with volume confirmation to avoid false signals.
# Target: 15-37 trades/year (60-150 over 4 years) using strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_rsi_pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-hour RSI(14)
    rsi = np.full(n, np.nan)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # 4-hour EMA(21) for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    
    if len(close_4h) >= 21:
        ema_4h[20] = np.mean(close_4h[:21])
        for i in range(21, len(close_4h)):
            ema_4h[i] = close_4h[i] * (2/22) + ema_4h[i-1] * (20/22)
    
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1-day volume average (20-period) for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_volume_1d = np.full(len(volume_1d), np.nan)
    
    for i in range(20, len(volume_1d)):
        avg_volume_1d[i] = np.mean(volume_1d[i-20:i])
    
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema_4h_val = ema_4h_aligned[i]
        avg_vol_1d_val = avg_volume_1d_aligned[i]
        vol_1d_idx = i // 24  # Approximate 1h to 1d index for volume check
        
        # Get current 1-day volume (approximate)
        if vol_1d_idx < len(df_1d):
            vol_1d = df_1d['volume'].iloc[vol_1d_idx] if hasattr(df_1d, 'iloc') else volume_1d[vol_1d_idx]
        else:
            vol_1d = 0
        
        if np.isnan(rsi_val) or np.isnan(ema_4h_val) or np.isnan(avg_vol_1d_val):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol_1d > 1.5 * avg_vol_1d_val
        
        if position == 1:  # Long position
            if rsi_val > 50:  # Exit when RSI crosses above 50
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            if rsi_val < 50:  # Exit when RSI crosses below 50
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Long setup: RSI oversold + 4h uptrend + volume surge
            if rsi_val < 30 and ema_4h_val > ema_4h_aligned[i-1] and vol_surge:
                position = 1
                signals[i] = 0.20
            # Short setup: RSI overbought + 4h downtrend + volume surge
            elif rsi_val > 70 and ema_4h_val < ema_4h_aligned[i-1] and vol_surge:
                position = -1
                signals[i] = -0.20
    
    return signals