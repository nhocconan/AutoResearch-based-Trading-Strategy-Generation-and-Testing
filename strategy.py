#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R1S1_Breakout_VolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakouts with 4h trend filter (EMA50) and 1d volume spike confirmation.
Long when price breaks above R1 in uptrend (4h EMA50 up + 1d volume > 1.5x 20-period avg).
Short when price breaks below S1 in downtrend (4h EMA50 down + 1d volume > 1.5x 20-period avg).
Session filter: 08-20 UTC to avoid low-liquidity hours.
Discrete sizing: 0.20. Target: 60-150 total trades over 4 years.
Works in bull/bear via 4h trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot points for 1h
    # Using previous bar's high, low, close
    phigh = np.roll(high, 1)
    plow = np.roll(low, 1)
    pclose = np.roll(close, 1)
    phigh[0] = high[0]
    plow[0] = low[0]
    pclose[0] = close[0]
    
    pivot = (phigh + plow + pclose) / 3
    range_ = phigh - plow
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    # 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume spike: volume > 1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    start_idx = max(50, 20)  # EMA50 and volume MA20 warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Session check
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike condition: 1d volume > 1.5x 20-period average
        volume_spike = volume_1d_aligned[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Trend conditions
        uptrend = ema_4h_aligned[i] > ema_4h_aligned[i-1]
        downtrend = ema_4h_aligned[i] < ema_4h_aligned[i-1]
        
        # Breakout conditions
        breakout_long = close[i] > r1[i] and close[i-1] <= r1[i-1]
        breakout_short = close[i] < s1[i] and close[i-1] >= s1[i-1]
        
        # Long logic: breakout above R1 + uptrend + volume spike
        if breakout_long and uptrend and volume_spike:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: breakout below S1 + downtrend + volume spike
        elif breakout_short and downtrend and volume_spike:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: long exits on break below S1 or trend change to downtrend
        elif position == 1 and (close[i] < s1[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        # Exit: short exits on break above R1 or trend change to uptrend
        elif position == -1 and (close[i] > r1[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_4h1d_Camarilla_R1S1_Breakout_VolumeSpike"
timeframe = "1h"
leverage = 1.0