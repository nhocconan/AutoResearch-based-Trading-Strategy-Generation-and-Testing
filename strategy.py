#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter.
# Long when price breaks above Donchian upper channel with 1d volume > 2.5x 30-period average and CHOP(14) > 61.8 (ranging market).
# Short when price breaks below Donchian lower channel with same volume and chop conditions.
# Exit when price reverts to Donchian midpoint (mean reversion in ranging markets).
# Uses discrete position sizing (0.25) to balance reward and risk while minimizing fee churn.
# Works in bull/bear: Donchian breakouts capture trends, chop filter ensures we only trade in ranging markets where mean reversion works.
# Volume spike confirms institutional participation in the breakout.

name = "4h_Donchian20_Breakout_1dVolumeSpike_ChopRegime"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF/Primary) ---
    # Donchian Channel (20 periods)
    donchian_hi = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lo = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_hi + donchian_lo) / 2
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: > 2.5x 30-period average (volume spike)
    vol_ma_30_1d = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values
    volume_confirm_1d = volume_1d > (2.5 * vol_ma_30_1d)
    
    # Choppiness Index (14 periods) - values > 61.8 indicate ranging market
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1).fillna(0).values
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop = np.zeros_like(close_1d)
    mask = range_14 > 0
    chop[mask] = 100 * np.log10(tr_sum_14[mask] / range_14[mask]) / np.log10(14)
    
    # Chop > 61.8 = ranging market (mean reversion regime)
    chop_regime = chop > 61.8
    
    # Align 1d indicators to 4h
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(donchian_hi[i]) or 
            np.isnan(donchian_lo[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + volume spike + chop regime (ranging)
            if (close[i] > donchian_hi[i] and 
                volume_confirm_1d_aligned[i] > 0.5 and
                chop_regime_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + volume spike + chop regime (ranging)
            elif (close[i] < donchian_lo[i] and 
                  volume_confirm_1d_aligned[i] > 0.5 and
                  chop_regime_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Donchian midpoint (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to Donchian midpoint (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals