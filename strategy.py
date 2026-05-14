#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w ADX trend filter and 1d volume confirmation.
# Long when price breaks above 20-day high with 1w ADX > 20 (trending) and 1d volume > 1.5x 20-period average.
# Short when price breaks below 20-day low with 1w ADX > 20 (trending) and 1d volume > 1.5x 20-period average.
# Exit on opposite 20-day level (20-day low for longs, 20-day high for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn. 1w ADX filter ensures trend alignment,
# reducing false breakouts in ranging markets. Volume confirmation adds momentum validation.
# Target: 30-100 total trades over 4 years = 7-25/year for 1d.
# Works in bull/bear: 1w ADX confirms trend strength, Donchian channels provide structure, volume confirms momentum.

name = "1d_Donchian20_Breakout_1wADX_1dVolumeConfirm"
timeframe = "1d"
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
    
    # --- 1d Indicators (LTF) ---
    # 1d volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # 1d Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w ADX(14)
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    tr = np.maximum(high_1w[1:] - low_1w[1:], 
                    np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), 
                               np.abs(low_1w[1:] - close_1w[:-1])))
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1w, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(adx_14_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + 1w ADX > 20 (trending) + volume confirmation
            if (close[i] > donchian_high[i] and 
                adx_14_aligned[i] > 20 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + 1w ADX > 20 (trending) + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  adx_14_aligned[i] > 20 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals