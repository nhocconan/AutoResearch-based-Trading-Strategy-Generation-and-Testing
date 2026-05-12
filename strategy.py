#!/usr/bin/env python3
name = "6h_TurtleSqueeze_1wTrend_VolumeFilter"
timeframe = "6h"
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
    
    # === 1W DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === TURTLE SQUEEZE (6h) ===
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Keltner channels (20-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = donchian_mid + (2.0 * atr)
    keltner_lower = donchian_mid - (2.0 * atr)
    
    # Squeeze condition: Bollinger Bands inside Keltner (low volatility)
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = donchian_mid + (2.0 * bb_std)
    bb_lower = donchian_mid - (2.0 * bb_std)
    squeeze = (bb_upper <= keltner_upper) & (bb_lower >= keltner_lower)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)  # High volume breakout
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_6h[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Squeeze breakout above Donchian high + weekly uptrend + volume spike
            if (squeeze[i-1] and  # was in squeeze
                close[i] > donchian_high[i] and 
                close[i] > ema50_1w_6h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Squeeze breakout below Donchian low + weekly downtrend + volume spike
            elif (squeeze[i-1] and  # was in squeeze
                  close[i] < donchian_low[i] and 
                  close[i] < ema50_1w_6h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below Donchian midpoint OR squeeze fires
            if close[i] < donchian_mid[i] or (not squeeze[i-1] and squeeze[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian midpoint OR squeeze fires
            if close[i] > donchian_mid[i] or (not squeeze[i-1] and squeeze[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals