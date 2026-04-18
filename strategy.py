#!/usr/bin/env python3
"""
4h_Donchian_Breakout_VolumeTrend_v2
Hypothesis: 4h Donchian(20) breakout with trend filter (EMA34) and volume confirmation.
Long when price breaks above upper Donchian with EMA34 up and volume > 1.5x average.
Short when price breaks below lower Donchian with EMA34 down and volume > 1.5x average.
Fixed position size 0.25. ATR-based stop loss via signal reversal.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drift.
Works in bull/bear via trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_up = ema_34 > np.roll(ema_34, 1)
    ema_34_down = ema_34 < np.roll(ema_34, 1)
    ema_34_up[0] = False
    ema_34_down[0] = False
    ema_34_up_aligned = align_htf_to_ltf(prices, df_1d, ema_34_up)
    ema_34_down_aligned = align_htf_to_ltf(prices, df_1d, ema_34_down)
    
    # Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter (optional, can be removed if too restrictive)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 34, 20)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_34_up_aligned[i]) or np.isnan(ema_34_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with uptrend and volume
            if close[i] > upper[i] and ema_34_up_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with downtrend and volume
            elif close[i] < lower[i] and ema_34_down_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below lower Donchian or trend changes
            if close[i] < lower[i] or not ema_34_up_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above upper Donchian or trend changes
            if close[i] > upper[i] or not ema_34_down_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeTrend_v2"
timeframe = "4h"
leverage = 1.0