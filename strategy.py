#!/usr/bin/env python3
# Hypothesis: 6h Williams %R combined with 1-day trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends (ADX > 25 on 1d),
# we fade extreme readings expecting continuation. In weak trends (ADX <= 25), we mean-revert.
# Volume spike confirms institutional participation. Designed to work in both bull and bear markets
# by adapting to regime. Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_WilliamsR_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 6h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max()
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # 1-day ADX (14-period) for trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up = high_1d.diff()
    down = -low_1d.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_dm = pd.Series(plus_dm, index=high_1d.index)
    minus_dm = pd.Series(minus_dm, index=high_1d.index)
    
    # Directional Indicators
    plus_di = 100 * plus_dm.rolling(window=14, min_periods=14).mean() / atr
    minus_di = 100 * minus_dm.rolling(window=14, min_periods=14).mean() / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_strong = adx_values > 25  # Strong trend
    
    # Align ADX to 6h
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong)
    
    # 1-day EMA (34) for trend direction
    ema_34 = close_1d.ewm(span=34, adjust=False, min_periods=34).mean()
    ema_34_values = ema_34.values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_values)
    
    # Volume spike (2x 20-period average) on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (2 * volume_ma)
    volume_spike = volume_spike.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warm-up period
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or
            np.isnan(adx_strong_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Strong trend regime (ADX > 25): fade extreme Williams %R
            if adx_strong_aligned[i]:
                # Fade oversold in uptrend (price above EMA34)
                if williams_r[i] < -80 and close[i] > ema_34_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Fade overbought in downtrend (price below EMA34)
                elif williams_r[i] > -20 and close[i] < ema_34_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # Weak trend regime (ADX <= 25): mean reversion at extremes
            else:
                # Oversold mean reversion
                if williams_r[i] < -80 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Overbought mean reversion
                elif williams_r[i] > -20 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral or trend weakens
            if williams_r[i] > -50 or not adx_strong_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral or trend weakens
            if williams_r[i] < -50 or not adx_strong_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals