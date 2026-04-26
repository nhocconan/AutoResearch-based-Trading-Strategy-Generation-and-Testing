#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation, filtered by choppiness regime.
Long when price breaks above upper Donchian channel in 1d uptrend with volume spike and chop > 61.8 (range).
Short when price breaks below lower Donchian channel in 1d downtrend with volume spike and chop > 61.8.
Donchian breakouts capture momentum, 1d trend filter avoids counter-trend trades, volume confirms strength,
and chop filter ensures we only trade in ranging markets where breakouts are meaningful.
Uses 12h primary timeframe with 1d HTF for trend/chop. Discrete position sizing (0.25) minimizes fee churn.
Targets 12-37 trades/year on 12h timeframe. Works in bull/bear by following 1d trend and avoids whipsaw in strong trends via chop filter.
"""

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
    
    # Get 1d data for trend filter and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Calculate 1d choppiness index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values / 14
    
    # Max/min close over 14 periods
    max_close = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (max_close - min_close)) / log10(14)
    chop = 100 * np.log10(atr_14 * 14 / (max_close - min_close + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Chop regime: > 61.8 = ranging (good for breakout mean reversion)
    chop_range = chop_aligned > 61.8
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    upper_donch = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_donch = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    upper_donch_aligned = align_htf_to_ltf(prices, df_12h, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_12h, lower_donch)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for Donchian/chop, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(upper_donch_aligned[i]) or np.isnan(lower_donch_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with 1d uptrend, chop range, and volume spike
            if (close[i] > upper_donch_aligned[i] and 
                uptrend_1d[i] and chop_range[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with 1d downtrend, chop range, and volume spike
            elif (close[i] < lower_donch_aligned[i] and 
                  downtrend_1d[i] and chop_range[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below lower Donchian OR 1d trend changes to downtrend OR chop leaves range
            if (close[i] < lower_donch_aligned[i] or not uptrend_1d[i] or not chop_range[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above upper Donchian OR 1d trend changes to uptrend OR chop leaves range
            if (close[i] > upper_donch_aligned[i] or not downtrend_1d[i] or not chop_range[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0