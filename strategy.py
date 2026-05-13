#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme reversal with 1d ADX trend filter and volume spike confirmation.
# Williams %R identifies overbought/oversold conditions. ADX > 25 confirms trend strength for continuation.
# Volume spike > 1.5x average validates institutional participation.
# Long: %R < -80 (oversold) AND ADX > 25 AND volume > 1.5 * volume_ma20
# Short: %R > -20 (overbought) AND ADX > 25 AND volume > 1.5 * volume_ma20
# Position size: 0.25 (discrete level to minimize fee churn). Target: 15-25 trades/year by requiring confluence of three filters.
# Works in bull markets via oversold bounces in uptrend, in bear markets via overbought reversals in downtrend.

name = "6h_WilliamsR_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for HTF ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data for trend filter
    # True Range
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr2 = np.absolute(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5 * 20-period MA
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Oversold (%R < -80) AND strong trend (ADX > 25) AND volume spike
            if williams_r[i] < -80 and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought (%R > -20) AND strong trend (ADX > 25) AND volume spike
            elif williams_r[i] > -20 and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Overbought (%R > -50) OR trend weakening (ADX < 20) OR volume dry up
            if williams_r[i] > -50 or adx_aligned[i] < 20 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Oversold (%R < -50) OR trend weakening (ADX < 20) OR volume dry up
            if williams_r[i] < -50 or adx_aligned[i] < 20 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals