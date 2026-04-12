#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_volume_v1"
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
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio (current ATR / 20-period ATR mean)
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate Camarilla levels using previous day's data
    camarilla_high = np.full(len(close_1d), np.nan)
    camarilla_low = np.full(len(close_1d), np.nan)
    pivot_point = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        camarilla_high[i] = C + ((H - L) * 1.1 / 2)
        camarilla_low[i] = C - ((H - L) * 1.1 / 2)
        pivot_point[i] = (H + L + C) / 3
    
    # Align Camarilla levels and pivot to 1h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    
    # Volume filter: current volume > 20-period average (on 1h data)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    # Position sizing: base size 0.20, scaled by ATR ratio (inverse volatility)
    vol_scaling = np.clip(1.0 / (atr_ratio_aligned + 0.001), 0.5, 1.5)
    position_size = 0.20 * vol_scaling
    position_size = np.clip(position_size, 0.10, 0.30)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(volume_ok[i]) or np.isnan(session_ok[i]) or
            np.isnan(position_size[i])):
            signals[i] = 0.0 if position == 0 else (position_size[i] if position == 1 else -position_size[i])
            continue
        
        # Breakout conditions with volume and session confirmation
        breakout_up = close[i] > camarilla_high_aligned[i]
        breakout_down = close[i] < camarilla_low_aligned[i]
        vol_ok = volume_ok[i]
        sess_ok = session_ok[i]
        
        # Entry signals
        long_signal = breakout_up and vol_ok and sess_ok
        short_signal = breakout_down and vol_ok and sess_ok
        
        # Exit when price returns to the Camarilla pivot (previous day's pivot)
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size[i]
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size[i]
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position with dynamic sizing
            if position == 1:
                signals[i] = position_size[i]
            elif position == -1:
                signals[i] = -position_size[i]
            else:
                signals[i] = 0.0
    
    return signals