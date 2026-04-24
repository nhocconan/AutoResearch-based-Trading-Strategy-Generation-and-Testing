#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1w ADX trend filter and volume spike confirmation.
- Donchian channels identify structural breakouts; 20-period balances sensitivity and reliability.
- 1w ADX > 25 ensures we only trade in strong trending markets (works in bull/bear regimes).
- Volume > 1.5x 50-bar average confirms institutional participation.
- Discrete position size 0.25 limits drawdown and reduces fee churn.
- Targets ~25-40 trades/year (100-160 total over 4 years) for fee efficiency.
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w ADX trend filter (uses completed weekly candles)
    high_1w = df_1w['high'].shift(1).values
    low_1w = df_1w['low'].shift(1).values
    close_1w = df_1w['close'].shift(1).values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # first value has no prior close
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align HTF indicators to LTF
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Donchian(20) breakout levels
    donchian_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_1w_aligned[i] > 25
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above Donchian high AND strong trend AND volume confirmation
            if close[i] > donchian_high[i] and strong_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND strong trend AND volume confirmation
            elif close[i] < donchian_low[i] and strong_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Donchian low OR loss of strong trend
            if close[i] < donchian_low[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above Donchian high OR loss of strong trend
            if close[i] > donchian_high[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1wADX_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0