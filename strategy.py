#!/usr/bin/env python3
# 1d_1w_donchian_volume_trend_v1
# Strategy: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture strong trends. Weekly trend filter ensures we only trade in the direction of the higher timeframe trend, reducing false signals in choppy markets. Volume confirmation adds conviction. Designed for low trade frequency (<25/year) to minimize fee decay in ranging markets, while capturing major moves in both bull and bear regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period high/low)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > high_max[i-1]  # Close above previous period's high
        breakout_down = close[i] < low_min[i-1]  # Close below previous period's low
        
        # Trend filter: price vs weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 20-period average
        vol_confirm = volume[i] > vol_avg_20[i]
        
        # Entry conditions
        # Long: upward breakout AND uptrend AND volume confirmation
        if breakout_up and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: downward breakout AND downtrend AND volume confirmation
        elif breakout_down and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout (contrarian signal)
        elif position == 1 and breakout_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals