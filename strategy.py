#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Adaptive_Turtle_Signal"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 1d ATR for volatility regime filter (same as Turtle)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d EMA for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h Donchian channels (20-period) - the core signal
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20, 55)  # Donchian, volume MA, EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh = donch_high[i]
        dl = donch_low[i]
        ema_trend = ema_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_ok = volume_filter[i]
        
        # Skip if ATR is too low (avoid choppy markets)
        if atr_val < 0.001 * close[i]:  # Avoid division by zero and noise
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Adaptive position size based on volatility (inverse volatility scaling)
        # Base size 0.25, scaled by inverse ATR (lower vol = larger position)
        vol_factor = np.clip(0.01 * close[i] / atr_val, 0.5, 2.0)  # Normalize around 1% ATR
        base_size = 0.25
        position_size = base_size * vol_factor
        position_size = np.clip(position_size, 0.15, 0.35)  # Keep within reasonable bounds
        
        if position == 0:
            # Enter long: price breaks above Donchian high + above EMA trend + volume
            if close[i] > dh and close[i] > ema_trend and vol_ok:
                signals[i] = position_size
                position = 1
            # Enter short: price breaks below Donchian low + below EMA trend + volume
            elif close[i] < dl and close[i] < ema_trend and vol_ok:
                signals[i] = -position_size
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend reversal
            if close[i] < dl or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend reversal
            if close[i] > dh or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals