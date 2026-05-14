#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ATR-normalized volume spike and ADX(14) trend filter on 4h.
# Uses Donchian channel (20-period high/low) for structure, ATR-scaled volume spike (>2.0x 20-bar mean) for conviction,
# and ADX > 25 on 4h to ensure trending markets. Discrete position sizing (0.0, ±0.30) to minimize fee churn.
# Designed to capture strong breakouts in trending markets while avoiding false signals in ranging conditions.
# Targets 20-50 trades/year per symbol.

name = "4h_Donchian20_Breakout_1dATRVolumeSpike_ADXFilter_v1"
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
    # ATR(14) for volatility normalization
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-scaled volume MA: 20-period average of volume / ATR
    vol_atr_ratio = volume / (atr_14 + 1e-10)
    vol_atr_ma_20 = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_atr_ratio > (2.0 * vol_atr_ma_20)
    
    # ADX (14) for trend strength on 4h
    plus_dm = np.where((high - high_shift) > (low_shift - low), np.maximum(high - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low) > (high - high_shift), np.maximum(low_shift - low, 0), 0)
    
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channel (20) from prior 1d bar
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h (wait for completed 1d bar)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(adx[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        if adx[i] <= 25:
            # In ranging/weak trend, stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Trending regime: look for breakouts with volume confirmation
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike
            if close[i] > donchian_high_20_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian low AND volume spike
            elif close[i] < donchian_low_20_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian low (mean reversion)
            if close[i] < donchian_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian high (mean reversion)
            if close[i] > donchian_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals