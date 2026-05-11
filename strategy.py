#!/usr/bin/env python3
name = "1h_4h1d_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # 4h volume filter: volume > 1.5x 20-period average
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    vol_4h = df_4h['volume'].values
    vol_ma20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma20_4h)
    volume_filter = volume > 1.5 * vol_ma20_4h_aligned
    
    # Calculate Camarilla levels from previous 4h bar
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Shift by 1 to use previous bar's data
    prev_typical = np.roll(typical_price, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # Set first value to NaN since we don't have previous bar
    prev_typical[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla R1, S1 levels
    R1 = prev_typical + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_typical - 1.1 * (prev_high - prev_low) / 12
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA, volume, and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma20_4h_aligned[i]) or
            np.isnan(R1[i]) or np.isnan(S1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above R1 + daily uptrend + volume filter
            if close[i] > R1[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: Close below S1 + daily downtrend + volume filter
            elif close[i] < S1[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close below S1 or daily trend down
            if close[i] < S1[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close above R1 or daily trend up
            if close[i] > R1[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals