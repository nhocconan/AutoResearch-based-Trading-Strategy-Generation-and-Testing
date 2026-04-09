#!/usr/bin/env python3
# 12h_donchian_volume_atr_v1
# Hypothesis: 12h Donchian breakout with volume confirmation and ATR filter.
# Long when price breaks above 20-bar Donchian high + volume > 1.5x 20-bar avg + ATR(14) > 0.01*close (avoid low-vol chop).
# Short when price breaks below 20-bar Donchian low + same volume/ATR conditions.
# Uses 1d HTF EMA(50) trend filter: only long if close > EMA50, short if close < EMA50.
# Discrete sizing: ±0.25. Target: 12-30 trades/year (50-120 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 14-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(volume_ma[i]) or
            np.isnan(atr[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # ATR filter: avoid extremely low volatility (chop)
        atr_filter = atr[i] > 0.01 * close[i]
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian low or volume/ATR conditions fail
            if close[i] <= low_min[i] or not (volume_confirmed and atr_filter):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
        # Exit: price reaches Donchian high or volume/ATR conditions fail
            if close[i] >= high_max[i] or not (volume_confirmed and atr_filter):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and atr_filter:
                # Long breakout: price breaks above Donchian high AND 1d trend filter (price > EMA50)
                if close[i] > high_max[i] and close[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price breaks below Donchian low AND 1d trend filter (price < EMA50)
                elif close[i] < low_min[i] and close[i] < ema_50_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals