#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION
# Hypothesis: Donchian channel breakout with volume confirmation and ATR-based stoploss works in both bull and bear markets.
# Long when price breaks above 20-period Donchian high with volume > 1.5x average volume.
# Short when price breaks below 20-period Donchian low with volume > 1.5x average volume.
# Exit when price returns to the middle of the Donchian channel or ATR-based stop is hit.
# Uses 1d EMA as trend filter to avoid counter-trend trades in strong trends.
# Targets 20-40 trades/year to minimize fee drain with high-probability setups.

name = "4H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION"
timeframe = "4h"
leverage = 1.0

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
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    pclose = df_1d['close'].values
    ema1d = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(atr[i]) or np.isnan(ema1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # LONG: price breaks above Donchian high with volume confirmation and uptrend
            if close[i] > highest_high[i] and vol_confirm and close[i] > ema1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low with volume confirmation and downtrend
            elif close[i] < lowest_low[i] and vol_confirm and close[i] < ema1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to Donchian mid or trend breaks
            if close[i] < donchian_mid[i] or close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to Donchian mid or trend breaks
            if close[i] > donchian_mid[i] or close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals