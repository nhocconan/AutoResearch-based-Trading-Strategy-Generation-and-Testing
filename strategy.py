#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# This strategy captures strong momentum moves by entering on price breaking above/below
# the 4-hour Donchian channel, filtered by daily EMA50 trend direction and volume spikes.
# Designed to work in both bull and bear markets by following the trend while avoiding
# false signals in low-volume environments. Target: 80-180 total trades over 4 years (20-45/year).
name = "4h_Donchian20_1dEMA50_VolumeConfirm"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channel (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50 = ema_50_1d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Price breaks above Donchian upper AND price > 1d EMA50 (uptrend) AND volume > 2x average
            if close[i] > upper and close[i] > ema_50 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Donchian lower AND price < 1d EMA50 (downtrend) AND volume > 2x average
            elif close[i] < lower and close[i] < ema_50 and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Donchian lower OR trend reverses (price < 1d EMA50)
            if close[i] < lower or close[i] < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above Donchian upper OR trend reverses (price > 1d EMA50)
            if close[i] > upper or close[i] > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals