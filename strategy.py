#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend.
- Donchian channel (20-period high/low) from prior completed 4h bar: Long when price > upper band, Short when price < lower band.
- Trend filter: Only trade in direction of 12h EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) from prior completed 4h bar (shifted by 1 to avoid look-ahead)
    # We calculate on prior completed bar, so we use rolling window on [0:i] but shift results by 1
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1  # EMA50 + Donchian20 + 1 for shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 12h EMA50 trend
            if i > 0 and not np.isnan(ema_50_12h_aligned[i-1]):
                ema50_slope = ema_50_12h_aligned[i] - ema_50_12h_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    if close[i] > donchian_high[i] and volume_spike[i]:
                        # Buy on Donchian breakout in uptrend
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    if close[i] < donchian_low[i] and volume_spike[i]:
                        # Sell on Donchian breakdown in downtrend
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price returns to Donchian middle or opposite break
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if not np.isnan(donchian_mid):
                if close[i] < donchian_mid or close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian middle or opposite break
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if not np.isnan(donchian_mid):
                if close[i] > donchian_mid or close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0