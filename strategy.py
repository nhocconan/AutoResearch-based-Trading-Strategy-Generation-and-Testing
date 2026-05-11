#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_4hTrend_1dVolume"
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
    
    # 4h trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    trend_4h_up = close > ema_20_4h_aligned
    
    # 1d volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_ma20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    volume_filter = volume > 1.5 * volume_ma20_1d_aligned
    
    # Camarilla levels (from previous day)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    signals = np.zeros(n)
    position = 0
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    start_idx = 20
    
    for i in range(start_idx, n):
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if np.isnan(trend_4h_up[i]) or np.isnan(volume_filter[i]) or np.isnan(R1[i]) or np.isnan(S1[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0 and in_session:
            # Long: price > R1, 4h uptrend, volume spike
            if close[i] > R1[i] and trend_4h_up[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price < S1, 4h downtrend, volume spike
            elif close[i] < S1[i] and not trend_4h_up[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < S1 or 4h downtrend
            if close[i] < S1[i] or not trend_4h_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > R1 or 4h uptrend
            if close[i] > R1[i] or trend_4h_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals