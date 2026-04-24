#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA50 trend.
- Donchian channel (20-period high/low) from prior 1d: Long when price > upper band, Short when price < lower band.
- Trend filter: Only trade in direction of 1w EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 1.5x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (we need prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need at least 20 periods for Donchian + 1 for prior bar
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels from prior 20 completed 1d bars
    # Upper band = max(high of last 20 days), Lower band = min(low of last 20 days)
    # We use rolling window on completed 1d bars, then shift by 1 to use prior day's values
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align to 1d timeframe (already aligned, just using the values)
    donchian_upper = high_20  # Already shifted for prior bar
    donchian_lower = low_20   # Already shifted for prior bar
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (using 1d volume data)
    # We need to get volume from 1d data for the MA calculation
    vol_1d = df_1d['volume'].values
    volume_ma = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + Donchian period
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1w EMA50 trend
            if i > 0 and not np.isnan(ema_50_1w_aligned[i-1]):
                ema50_slope = ema_50_1w_aligned[i] - ema_50_1w_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    if close[i] > donchian_upper[i] and volume_spike[i]:
                        # Buy on Donchian breakout in uptrend
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    if close[i] < donchian_lower[i] and volume_spike[i]:
                        # Sell on Donchian breakdown in downtrend
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price returns to Donchian lower band or opposite break
            if not np.isnan(donchian_lower[i]) and not np.isnan(donchian_upper[i]):
                if close[i] < donchian_lower[i] or close[i] > donchian_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian upper band or opposite break
            if not np.isnan(donchian_upper[i]) and not np.isnan(donchian_lower[i]):
                if close[i] > donchian_upper[i] or close[i] < donchian_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0