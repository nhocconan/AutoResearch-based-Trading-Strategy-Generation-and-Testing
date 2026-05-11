#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_up = close > ema_34_1d_aligned
    
    # 1d volume filter: volume > 1.8x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.8 * vol_ma20_1d_aligned
    
    # Camarilla levels (H4/L4) from previous 1d
    h4_prev = df_1d['high'].values
    l4_prev = df_1d['low'].values
    close_prev = df_1d['close'].values
    # Camarilla H4 = Close + 1.1 * (High - Low) / 2
    # Camarilla L4 = Close - 1.1 * (High - Low) / 2
    camarilla_h4 = close_prev + 1.1 * (h4_prev - l4_prev) / 2
    camarilla_l4 = close_prev - 1.1 * (h4_prev - l4_prev) / 2
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
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
            # Long: Close above Camarilla H4 + daily uptrend + volume filter
            if close[i] > camarilla_h4_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: Close below Camarilla L4 + daily downtrend + volume filter
            elif close[i] < camarilla_l4_aligned[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Close below Camarilla L4 or daily trend down
            if close[i] < camarilla_l4_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Close above Camarilla H4 or daily trend up
            if close[i] > camarilla_h4_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals