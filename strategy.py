#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume spike confirmation.
- Long when price breaks above 4h Donchian upper band AND 1d close > 1d EMA50 AND volume > 2.0 * 20-period average
- Short when price breaks below 4h Donchian lower band AND 1d close < 1d EMA50 AND volume > 2.0 * 20-period average
- Exit on opposite Donchian band (exit long on lower band, exit short on upper band)
- Uses 4h primary with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Donchian provides clear trend structure; EMA50 filters regime; volume spike confirms momentum
- Designed to work in both bull (breakouts with trend) and bear (breakouts against trend) markets
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1d_aligned
    bearish_regime = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 2.0 * 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20) + 1  # Need Donchian, EMA50, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian upper AND bullish regime AND volume confirmation
            if close[i] > donchian_upper[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower AND bearish regime AND volume confirmation
            elif close[i] < donchian_lower[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian lower (opposite band)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian upper (opposite band)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0