#!/usr/bin/env python3
"""
1h_Donchian_Breakout_4hTrend_Volume
Hypothesis: 1h price breaking out of 4h Donchian channel with 1d trend filter and volume spike captures momentum with controlled frequency. Uses 4h for trend/direction, 1h for entry timing to target 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (20-period high/low)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h (wait for 4h bar close)
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5 * 24-period average (1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for Donchian (20) and EMA (50)
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]) or 
            np.isnan(ema50_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        upper = donchian_high_1h[i]
        lower = donchian_low_1h[i]
        ema_trend = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with uptrend and volume spike
            if close[i] > upper and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below 4h Donchian low with downtrend and volume spike
            elif close[i] < lower and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below 4h Donchian low or trend turns down
            if close[i] < lower or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above 4h Donchian high or trend turns up
            if close[i] > upper or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Donchian_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0