#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volatility_filter_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly ATR and volatility
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR (14-period)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly ATR ratio (current ATR / 20-period ATR mean)
    atr_ma_20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1w / atr_ma_20
    
    # Align ATR ratio to 1d timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Calculate ATR-based position size (inverse volatility scaling)
    # Higher volatility = smaller position, capped at 0.30
    vol_scaling = np.clip(1.0 / (atr_ratio_aligned + 0.001), 0.5, 1.5)  # Scale between 0.5x and 1.5x
    base_size = 0.25
    position_size = base_size * vol_scaling
    position_size = np.clip(position_size, 0.10, 0.30)  # Keep within reasonable bounds
    
    # Get 1d data for Camarilla calculation (using previous day's data)
    camarilla_high = np.full(len(close), np.nan)
    camarilla_low = np.full(len(close), np.nan)
    
    for i in range(1, len(close)):
        H = high[i-1]
        L = low[i-1]
        C = close[i-1]
        camarilla_high[i] = C + ((H - L) * 1.1 / 2)
        camarilla_low[i] = C - ((H - L) * 1.1 / 2)
    
    # Volume filter: current volume > 20-period average (on 1d data)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # warmup
        # Skip if not ready
        if (np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or 
            np.isnan(volume_ok[i]) or np.isnan(position_size[i])):
            signals[i] = 0.0 if position == 0 else (position_size[i] if position == 1 else -position_size[i])
            continue
        
        # Breakout conditions with volume confirmation
        breakout_up = close[i] > camarilla_high[i]
        breakout_down = close[i] < camarilla_low[i]
        vol_ok = volume_ok[i]
        
        # Entry signals
        long_signal = breakout_up and vol_ok
        short_signal = breakout_down and vol_ok
        
        # Exit when price returns to the Camarilla pivot (close of previous day)
        pivot_point = np.full(len(close), np.nan)
        for j in range(1, len(close)):
            H = high[j-1]
            L = low[j-1]
            C = close[j-1]
            pivot_point[j] = (H + L + C) / 3
        
        exit_long = close[i] < pivot_point[i]
        exit_short = close[i] > pivot_point[i]
        
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