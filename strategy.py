#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Uses Donchian channel from primary 6h for structure, 12h EMA50 for trend direction,
# and 6h volume spike for conviction. Designed to capture strong breakouts in trending
# markets while avoiding counter-trend trades. Discrete position sizing (0.0, ±0.25)
# minimizes fee churn. Targets 12-37 trades/year per symbol.

name = "6h_Donchian20_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # Donchian Channel (20)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # EMA 50 on 12h
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if missing data
        if (np.isnan(donchian_high_20[i]) or
            np.isnan(donchian_low_20[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in direction of 12h trend
        if close[i] > ema_50_12h_aligned[i]:
            # Uptrend: look for long breakouts
            if position == 0:
                # LONG: Price breaks above Donchian high AND volume spike
                if close[i] > donchian_high_20[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Stay long unless reversal
                signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: Price above EMA (trend changed) OR Donchian breakout
                if close[i] > donchian_high_20[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = -0.25
        else:
            # Downtrend: look for short breakouts
            if position == 0:
                # SHORT: Price breaks below Donchian low AND volume spike
                if close[i] < donchian_low_20[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == -1:
                # Stay short unless reversal
                signals[i] = -0.25
            elif position == 1:
                # EXIT LONG: Price below EMA (trend changed) OR Donchian breakout
                if close[i] < donchian_low_20[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.25
    
    return signals