#/usr/bin/env python3
name = "12h_24hBreakout_1dTrend_VolumeFilter_v1"
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
    
    # Get daily data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA34 trend filter (requires 34 periods)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 24-hour high/low breakout (2 periods of 12h)
    high_24h = pd.Series(high).rolling(window=2, min_periods=2).max().shift(1).values
    low_24h = pd.Series(low).rolling(window=2, min_periods=2).min().shift(1).values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(2, 20)  # Need 2 for 24h breakout, 20 for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 24h high AND above daily EMA34 + volume
            if close[i] > high_24h[i] and close[i] > ema_34_1d_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 24h low AND below daily EMA34 + volume
            elif close[i] < low_24h[i] and close[i] < ema_34_1d_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to 24h range or breaks in opposite direction
            if position == 1:
                if close[i] < low_24h[i] or close[i] < ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_24h[i] or close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals