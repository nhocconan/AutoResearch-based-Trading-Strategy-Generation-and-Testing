#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA trend.
- Donchian channels: Upper band = 20-period high, Lower band = 20-period low from prior 1d OHLC.
- Breakout: Close > Upper band (long) or Close < Lower band (short) with volume > 2.0x 20-period volume MA.
- Trend filter: Only trade breakouts in direction of 1w EMA50 (long if close > EMA50, short if close < EMA50).
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get prior 1d OHLC for Donchian channels (shifted by 1 to avoid look-ahead)
    # We need to use completed 1d bars only, so we shift the 1d data by 1 bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels from prior 1d data (20-period high/low)
    # Use rolling window on completed 1d bars, then align to 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (no additional delay needed as we used completed bars)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 1d timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Donchian breakout with volume spike and trend filter
            if volume_spike[i]:
                # Long breakout: close > Donchian high and close > 1w EMA50 (uptrend)
                if close[i] > donchian_high_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < Donchian low and close < 1w EMA50 (downtrend)
                elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Donchian channel or opposite signal
            if close[i] < donchian_low_aligned[i]:  # Exit when price falls below Donchian low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Donchian channel or opposite signal
            if close[i] > donchian_high_aligned[i]:  # Exit when price rises above Donchian high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0