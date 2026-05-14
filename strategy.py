#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d ADX(14) > 25 trend filter and 6h volume > 1.5x 20-period average confirmation.
# Long when price breaks above 20-period high with ADX > 25 (strong trend) and volume spike.
# Short when price breaks below 20-period low with ADX > 25 and volume spike.
# Exit on opposite Donchian level or when ADX < 20 (trend weakening).
# Uses Donchian channels for structure, ADX for trend strength (works in both bull/bear by filtering only strong trends),
# and volume to confirm participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian20_Breakout_1dADX_6hVolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) - trend strength filter
    # Calculate True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate +DI and -DI
    up_move = pd.Series(high_1d).diff()
    down_move = -(pd.Series(low_1d).diff())
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di_14 = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_14)
    
    # Calculate ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(high_ma_20[i]) or
            np.isnan(low_ma_20[i]) or
            np.isnan(volume_spike_6h[i]) or
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-period high + ADX > 25 (strong trend) + volume spike
            if (close[i] > high_ma_20[i] and 
                adx_14_aligned[i] > 25 and 
                volume_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low + ADX > 25 (strong trend) + volume spike
            elif (close[i] < low_ma_20[i] and 
                  adx_14_aligned[i] > 25 and 
                  volume_spike_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low OR ADX < 20 (trend weakening)
            if (close[i] < low_ma_20[i] or adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high OR ADX < 20 (trend weakening)
            if (close[i] > high_ma_20[i] or adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals