#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio (current ATR / 20-period ATR mean)
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / atr_ma_20
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate ATR-based position size (inverse volatility scaling)
    # Higher volatility = smaller position, capped at 0.30
    vol_scaling = np.clip(1.0 / (atr_ratio_aligned + 0.001), 0.5, 1.5)
    base_size = 0.25
    position_size = base_size * vol_scaling
    position_size = np.clip(position_size, 0.10, 0.30)
    
    # Calculate Donchian channel (20-period high/low) using previous day's data
    donchian_high = np.full(len(close_1d), np.nan)
    donchian_low = np.full(len(close_1d), np.nan)
    
    for i in range(20, len(close_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: current volume > 20-period average (on 4h data)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ok[i]) or np.isnan(position_size[i])):
            signals[i] = 0.0 if position == 0 else (position_size[i] if position == 1 else -position_size[i])
            continue
        
        # Breakout conditions with volume confirmation
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        vol_ok = volume_ok[i]
        
        # Entry signals
        long_signal = breakout_up and vol_ok
        short_signal = breakout_down and vol_ok
        
        # Exit when price returns to the opposite Donchian level
        exit_long = close[i] < donchian_low_aligned[i]
        exit_short = close[i] > donchian_high_aligned[i]
        
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