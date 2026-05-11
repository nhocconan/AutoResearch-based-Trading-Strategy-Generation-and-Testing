#!/usr/bin/env python3
name = "1h_Supertrend_Trend_4h_1d_Confirm"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend: Supertrend (ATR-based)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # ATR for Supertrend
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic upper and lower bands
    hl2 = (high_4h + low_4h) / 2
    upper = hl2 + 3 * atr
    lower = hl2 - 3 * atr
    
    # Supertrend calculation
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    for i in range(1, len(close_4h)):
        if close_4h[i] > supertrend[i-1]:
            supertrend[i] = max(lower[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
            direction[i] = -1
    
    # Align 4h Supertrend direction to 1h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # 1d volume spike: current volume > 2x 24-period average (on 1h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 2)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend_dir_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(in_session[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if in_session[i]:
            if position == 0:
                # Long: 4h uptrend AND volume spike
                if supertrend_dir_aligned[i] == 1 and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: 4h downtrend AND volume spike
                elif supertrend_dir_aligned[i] == -1 and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
            elif position == 1:
                # Long exit: 4h downtrend
                if supertrend_dir_aligned[i] == -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20  # maintain position
            elif position == -1:
                # Short exit: 4h uptrend
                if supertrend_dir_aligned[i] == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20  # maintain position
        else:
            # Outside session: flatten
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals