#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: Trade 6h Donchian(20) breakouts with 12h EMA50 trend filter and volume confirmation. 
Donchian breakouts capture momentum, 12h EMA50 ensures alignment with intermediate trend, 
volume spike confirms institutional participation. Designed to work in bull markets (breakouts with trend) 
and bear markets (mean reversion at extremes with trend filter). Targets 50-150 total trades over 4 years 
(12-37/year) to avoid fee drag. Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 6h
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 12h EMA(50), ATR(14), Donchian(20), volume MA(20)
    start_idx = max(50, 14, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_12h_up = close_val > ema_50_12h_aligned[i]   # 12h uptrend
        trend_12h_down = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 12h trend up AND volume spike
            long_signal = (close_val > donchian_upper[i]) and trend_12h_up and vol_spike
            
            # Short: price breaks below Donchian lower AND 12h trend down AND volume spike
            short_signal = (close_val < donchian_lower[i]) and trend_12h_down and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions: trend flips down OR stoploss hit
            if (not trend_12h_up) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions: trend flips up OR stoploss hit
            if (not trend_12h_down) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0