#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeRegime_v1
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume regime filter (volume > 1.5x 20-period mean). 
Enters long when price breaks above upper Donchian + 12h uptrend + volume regime; enters short when price breaks below lower Donchian + 12h downtrend + volume regime. 
Exits on opposite Donchian break or trend reversal. Uses discrete position sizing (0.25) to minimize fee churn. 
Volume regime filter avoids low-volume false breakouts. Trend filter ensures alignment with higher timeframe momentum. 
Designed for 75-150 trades over 4 years (~19-38/year) to stay within fee drag limits. Works in bull/bear via trend following logic.
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = invalid
    trend_12h = np.where(ema_50_12h_aligned > 0, 
                         np.where(close > ema_50_12h_aligned, 1, -1), 
                         0)
    
    # Donchian(20) channels
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: volume > 1.5 * 20-period mean
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    # Start after warmup (need 50 for EMA, 20 for Donchian and volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_h[i]) or 
            np.isnan(donchian_l[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(trend_12h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            bars_since_entry += 1
            continue
        
        # Donchian breakout conditions with trend and volume regime confirmation
        if position == 0:
            # Long: Price breaks above upper Donchian AND 12h uptrend AND volume regime
            if close[i] > donchian_h[i] and trend_12h[i] == 1 and volume_regime[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price breaks below lower Donchian AND 12h downtrend AND volume regime
            elif close[i] < donchian_l[i] and trend_12h[i] == -1 and volume_regime[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: Price falls below lower Donchian OR 12h trend turns down
            if close[i] < donchian_l[i] or trend_12h[i] == -1:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: Price rises above upper Donchian OR 12h trend turns up
            if close[i] > donchian_h[i] or trend_12h[i] == 1:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeRegime_v1"
timeframe = "4h"
leverage = 1.0