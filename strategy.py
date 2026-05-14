#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ADX trend filter.
# Uses Donchian channel for structure, 12h volume spike for conviction, and ADX(14) > 25 to ensure trending markets.
# Discrete position sizing (0.0, ±0.30) minimizes fee churn. Designed to capture strong breakouts in trending markets
# while avoiding false signals in ranging conditions. Targets 20-40 trades/year per symbol.

name = "4h_Donchian20_Breakout_12hVolumeSpike_ADXFilter_v1"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # Donchian Channel (20)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX(14) for trend strength
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # +DI and -DI
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    dx = np.nan_to_num(dx, nan=0.0)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Volume spike: > 2.0x 20-period average
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (2.0 * vol_ma_20_12h)
    
    # Align to 4h (wait for completed 12h bar)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(donchian_high_20[i]) or
            np.isnan(donchian_low_20[i]) or
            np.isnan(adx_14[i]) or
            np.isnan(volume_spike_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        if adx_14[i] <= 25:
            # No trend, stay flat
            signals[i] = 0.0
            continue
        
        # Trending regime: look for breakouts
        if position == 0:
            # LONG: Price breaks above Donchian high AND 12h volume spike
            if close[i] > donchian_high_20[i] and volume_spike_12h_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian low AND 12h volume spike
            elif close[i] < donchian_low_20[i] and volume_spike_12h_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low (stoploss) or loses volume momentum
            if close[i] < donchian_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high (stoploss) or loses volume momentum
            if close[i] > donchian_high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals