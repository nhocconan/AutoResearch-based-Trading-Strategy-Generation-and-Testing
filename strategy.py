#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1
Hypothesis: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation (2.0x) to capture strong trending moves in BTC/ETH/SOL. 
Works in bull via upside breakouts in uptrend, works in bear via downside breakouts in downtrend. 
Low frequency (~10-25 trades/year) minimizes fee drag. Uses discrete sizing (0.30) for stability.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = invalid
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
                        0)
    
    # Calculate Donchian(20) channels from daily OHLC (using 20-day lookback)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2.0 * volume_ma(50) for strong confirmation
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for Donchian, 50 for volume MA)
    start_idx = max(50, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i]) or
            np.isnan(trend_1w[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Donchian breakout conditions with trend and volume filters
        if position == 0:
            # Long: Price breaks above 20-day high AND 1w uptrend AND volume spike (2.0x)
            if close[i] > highest_high[i] and trend_1w[i] == 1 and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below 20-day low AND 1w downtrend AND volume spike (2.0x)
            elif close[i] < lowest_low[i] and trend_1w[i] == -1 and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: Price falls below 20-day low OR 1w trend turns down
            if close[i] < lowest_low[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: Price rises above 20-day high OR 1w trend turns up
            if close[i] > highest_high[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0