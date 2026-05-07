#!/usr/bin/env python3
name = "1d_KAMA_1wTrend_VolumeFilter_v1"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA on daily close
    close_s = pd.Series(close)
    change = close_s.diff(10).abs()
    volatility = close_s.diff().abs().rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # Weekly EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(10, 20)  # Need 10 for KAMA, 20 for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA AND above weekly EMA34 + volume
            if close[i] > kama[i] and close[i] > ema_34_1w_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA AND below weekly EMA34 + volume
            elif close[i] < kama[i] and close[i] < ema_34_1w_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to KAMA or breaks in opposite direction
            if position == 1:
                if close[i] < kama[i] or close[i] < ema_34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama[i] or close[i] > ema_34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals