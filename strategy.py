#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_ATRRegime_VolumeSpike
Hypothesis: Trade 12h Donchian(20) breakouts in direction of 1d trend with ATR-based regime filter (low volatility = range, high volatility = trend) and volume spike confirmation.
In bull markets: trend filter + volatility expansion captures strong moves.
In bear markets: regime filter avoids false breakouts in low-volatility chop, volume spike confirms institutional interest.
Discrete sizing 0.25 to manage risk and minimize fee churn. Target: 15-30 trades/year.
"""

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
    
    # Get daily data for trend filter and ATR regime
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_30)
    
    # ATR regime: current ATR > 1.2x 50-period average (volatility expansion = trend regime)
    atr_12h = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_12h).rolling(window=50, min_periods=50).mean().values
    atr_regime = atr_12h > (1.2 * atr_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for daily EMA50 (50), daily ATR (14+50), Donchian (20), volume MA (30), ATR regime (14+50)
    start_idx = max(50, 64, 20, 30, 64)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_30[i]) or np.isnan(atr_ma_50[i]) or
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND daily trend bullish AND volatility expansion regime AND volume spike
            long_setup = (close[i] > donchian_high[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         atr_regime[i] and \
                         volume_spike[i]
            # Short: price breaks below Donchian low AND daily trend bearish AND volatility expansion regime AND volume spike
            short_setup = (close[i] < donchian_low[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          atr_regime[i] and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel OR daily trend turns bearish OR volatility contracts (range regime)
            if (close[i] < donchian_high[i] and close[i] > donchian_low[i]) or \
               (close[i] < ema_50_1d_aligned[i]) or \
               (not atr_regime[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR daily trend turns bullish OR volatility contracts (range regime)
            if (close[i] < donchian_high[i] and close[i] > donchian_low[i]) or \
               (close[i] > ema_50_1d_aligned[i]) or \
               (not atr_regime[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_ATRRegime_VolumeSpike"
timeframe = "12h"
leverage = 1.0