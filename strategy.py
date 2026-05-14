#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme combined with 1d ADX regime filter and volume spike confirmation.
# Williams %R < -80 = oversold (long), > -20 = overbought (short). Only trade in strong trends (ADX > 25)
# to avoid whipsaws in ranging markets. Volume spike (>2.0x 20-period average) confirms momentum.
# Designed to capture trend continuation moves in both bull and bear markets by trading extremes
# only when the 1d trend is strong. Targets 50-150 total trades over 4 years.

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike_v2"
timeframe = "6h"
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
    
    # Williams %R(14) - momentum oscillator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-9) * -100
    
    # Volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) - trend strength filter
    # ADX calculation requires +DI and -DI
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = abs(pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    up_move = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    down_move = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    plus_di = 100 * (pd.Series(up_move).rolling(window=14, min_periods=14).mean() / atr_1d)
    minus_di = 100 * (pd.Series(down_move).rolling(window=14, min_periods=14).mean() / atr_1d)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to LTF
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)  # Williams %R needs no extra delay
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)          # ADX needs no extra delay
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))  # Wait for 1d close
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND ADX > 25 (strong trend) AND volume spike
            if (williams_r_aligned[i] < -80 and 
                adx_1d_aligned[i] > 25 and 
                volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND ADX > 25 (strong trend) AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (return from oversold) OR ADX < 20 (trend weakens)
            if williams_r_aligned[i] > -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (return from overbought) OR ADX < 20 (trend weakens)
            if williams_r_aligned[i] < -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals