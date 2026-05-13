#!/usr/bin/env python3
"""
1h_4hDonchian20_1dTrend_Filter_v1
Hypothesis: Breakouts above 4h Donchian(20) high or below low with 1d EMA50 trend filter, volume > 1.5x 20-period avg, and session filter (08-20 UTC) to reduce noise. Designed for 1h to capture trend moves with controlled trade frequency (~20-50/year) in both bull and bear markets.
"""

name = "1h_4hDonchian20_1dTrend_Filter_v1"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low) on 4h
    donch_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Only allow entries during session
        if not session_filter[i]:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 4h Donchian high with volume confirmation in uptrend
            if donch_high_aligned[i] > 0 and not np.isnan(donch_high_aligned[i]) and \
               high[i] > donch_high_aligned[i] and volume_confirmed[i] and \
               close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below 4h Donchian low with volume confirmation in downtrend
            elif donch_low_aligned[i] > 0 and not np.isnan(donch_low_aligned[i]) and \
                 low[i] < donch_low_aligned[i] and volume_confirmed[i] and \
                 close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below Donchian low or trend weakens
            if donch_low_aligned[i] > 0 and not np.isnan(donch_low_aligned[i]) and \
               (low[i] < donch_low_aligned[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses back above Donchian high or trend weakens
            if donch_high_aligned[i] > 0 and not np.isnan(donch_high_aligned[i]) and \
               (high[i] > donch_high_aligned[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals