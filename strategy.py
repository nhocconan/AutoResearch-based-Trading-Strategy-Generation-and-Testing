#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Uses Donchian channels for structure, 1d EMA34 for trend direction, and volume spike for conviction.
# Discrete position sizing (0.0, ±0.30) minimizes fee churn. Designed to capture strong breakouts
# in trending markets while avoiding false signals in ranging conditions. Targets 20-50 trades/year.

name = "4h_Donchian20_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # --- 4h Indicators (LTF) ---
    # Donchian Channel (20) - upper/lower bounds
    donch_h_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_l_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: > 1.8x 20-period average (slightly looser to increase trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # EMA34 on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after Donchian warmup
        # Skip if missing data
        if (np.isnan(donch_h_20[i]) or
            np.isnan(donch_l_20[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # LONG: Price breaks above Donchian HIGH + above 1d EMA34 + volume spike
        if close[i] > donch_h_20[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = 0.30
        # SHORT: Price breaks below Donchian LOW + below 1d EMA34 + volume spike
        elif close[i] < donch_l_20[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = -0.30
        # EXIT: Price reverses back into Donchian channel (mean reversion)
        elif position == 1 and close[i] < donch_h_20[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donch_l_20[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals