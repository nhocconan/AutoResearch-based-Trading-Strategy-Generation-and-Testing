#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    trend_4h_up = close > ema_50_4h_aligned
    
    # 1d volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_spike = volume > 1.5 * vol_ma20_1d_aligned
    
    # 1h Camarilla levels (R1, S1)
    # Previous hour's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    range_hl = prev_high - prev_low
    
    # Camarilla R1 and S1
    R1 = prev_close + range_hl * 1.1 / 12
    S1 = prev_close - range_hl * 1.1 / 12
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # ensure indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(trend_4h_up[i]) or np.isnan(volume_spike[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1, 4h uptrend, volume spike, session
            if close[i] > R1[i] and trend_4h_up[i] and volume_spike[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, 4h downtrend, volume spike, session
            elif close[i] < S1[i] and not trend_4h_up[i] and volume_spike[i] and session_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 or 4h trend turns down
            if close[i] < S1[i] or not trend_4h_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R1 or 4h trend turns up
            if close[i] > R1[i] or trend_4h_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals