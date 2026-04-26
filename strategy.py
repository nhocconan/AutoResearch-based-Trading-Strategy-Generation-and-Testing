#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRVolRegime
Hypothesis: Donchian(20) breakouts with ATR-based volatility regime filter and volume confirmation capture strong trending moves while avoiding whipsaws in choppy markets. Uses 1d HTF for trend alignment and volatility regime detection. Designed for 4h timeframe with tight entry conditions to limit trades to 20-50/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter (more responsive than EMA50)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Donchian(20) channels on 4h
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # ATR(14) on 4h for dynamic stop and volatility filter
    tr_4h1 = np.abs(high - low)
    tr_4h2 = np.abs(high - np.roll(close, 1))
    tr_4h3 = np.abs(low - np.roll(close, 1))
    tr_4h1[0] = np.nan
    tr_4h2[0] = np.nan
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume spike detector (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Volatility regime filter: trade only when 1d ATR is above its 50-period MA (avoid low volatility chop)
    atr_ma_50_1d = pd.Series(atr_14_1d_aligned).rolling(window=50, min_periods=50).mean().values
    vol_regime = atr_14_1d_aligned > atr_ma_50_1d  # High volatility regime = trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(50, 20)  # 1d EMA34 and Donchian20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_14_4h[i]) or np.isnan(volume_spike[i]) or np.isnan(vol_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume spike, 1d uptrend, and high volatility regime
            if close[i] > donchian_high[i] and volume_spike[i] and uptrend and vol_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume spike, 1d downtrend, and high volatility regime
            elif close[i] < donchian_low[i] and volume_spike[i] and downtrend and vol_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price re-enters Donchian channel (below midpoint) OR 1d trend changes to downtrend OR volatility drops (low regime)
            if close[i] < donchian_mid[i] or not uptrend or not vol_regime[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price re-enters Donchian channel (above midpoint) OR 1d trend changes to uptrend OR volatility drops (low regime)
            if close[i] > donchian_mid[i] or not downtrend or not vol_regime[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_ATRVolRegime"
timeframe = "4h"
leverage = 1.0