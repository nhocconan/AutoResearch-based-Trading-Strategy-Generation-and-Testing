#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian(20) breakout with 4h/1d trend filters and volume confirmation.
- Long when price breaks above Donchian upper band and close > 4h EMA20 AND 1d EMA34 (bullish alignment)
- Short when price breaks below Donchian lower band and close < 4h EMA20 AND 1d EMA34 (bearish alignment)
- Volume must be > 1.8x 20-period average for high-conviction breakouts
- ATR(14) trailing stop: exit when price moves 2.0x ATR from extreme since entry
- Uses 4h/1d HTF for trend alignment (proven effective across multiple experiments)
- Session filter: only trade 08-20 UTC to reduce noise
- Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
- Discrete position sizing: 0.20 to control drawdown
"""

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
    open_time = prices['open_time'].values
    
    # Calculate Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # Get 4h data ONCE before loop for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.8x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    # ATR(14) for volatility and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 34, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, trend up (close > 4h EMA20 AND 1d EMA34), volume spike
            if close[i] > donchian_upper[i] and close[i] > ema_20_4h_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below Donchian lower, trend down (close < 4h EMA20 AND 1d EMA34), volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema_20_4h_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Long exit: price drops 2.0x ATR from highest high since entry
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Short exit: price rises 2.0x ATR from lowest low since entry
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hEMA20_1dEMA34_VolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0