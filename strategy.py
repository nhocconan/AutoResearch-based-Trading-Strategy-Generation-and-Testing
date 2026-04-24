#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h, HTF: 1d for EMA34 trend alignment (proven BTC/ETH edge).
- Donchian breakout: long when price closes above upper channel, short when below lower channel.
- Trend filter: only long when 12h close > 1d EMA34, only short when 12h close < 1d EMA34.
- Volume confirmation: current 12h volume > 1.8 * 20-period 12h volume MA (moderate to avoid overtrading).
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) from prior 12h OHLC
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Align 12h data to primary timeframe (same as prices since primary=12h)
    # For 12h primary timeframe, we can use the 12h data directly with proper indexing
    # But we need to align it to the 12h bars in prices
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Donchian channels: upper = max(high_12h over 20 periods), lower = min(low_12h over 20 periods)
    high_series = pd.Series(high_12h_aligned)
    low_series = pd.Series(low_12h_aligned)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    # Trend filter: 12h close vs 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34, 20)  # Need sufficient lookback for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above Donchian upper AND uptrend AND volume spike
            if close[i] > donchian_upper[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below Donchian lower AND downtrend AND volume spike
            elif close[i] < donchian_lower[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint of Donchian channels or reverse signal
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] <= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to midpoint of Donchian channels or reverse signal
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] >= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0