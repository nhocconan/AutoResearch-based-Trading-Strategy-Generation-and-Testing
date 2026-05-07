#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume_v1
Hypothesis: Use 4h trend (EMA34) and 1d volume confirmation to filter 1h Camarilla breakouts.
Limits trades to 15-37/year via strict conditions: price must break R1/S1, align with 4h trend,
and have above-average 1d volume. Works in bull/bear via trend filter.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume_v1"
timeframe = "1h"
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
    
    # Previous 1h bar for Camarilla levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    rng = prev_high - prev_low
    R1 = prev_close + rng * 1.12
    S1 = prev_close - rng * 1.12
    
    # 4h trend: EMA34 of 4h close
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume: average of daily volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(ema_4h_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or vol_ma_1d_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0 and in_session:
            # Long: break above R1, above 4h EMA, high 1d volume
            if (close[i] > R1[i] and close[i] > ema_4h_aligned[i] and
                volume[i] > vol_ma_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below S1, below 4h EMA, high 1d volume
            elif (close[i] < S1[i] and close[i] < ema_4h_aligned[i] and
                  volume[i] > vol_ma_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: close crosses below S1 (mean reversion)
            if close[i] < S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: close crosses above R1 (mean reversion)
            if close[i] > R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals