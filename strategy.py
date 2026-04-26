#!/usr/bin/env python3
"""
1h_4hDonchian20_1dTrend_SessionVolume_v1
Hypothesis: 1h timeframe with 4h Donchian(20) breakout for entry timing, 
1d EMA50 for trend filter, and session filter (08-20 UTC) to reduce noise.
Uses volume confirmation to avoid false breakouts. 
Designed for low trade frequency (target 15-37/year) to minimize fee drag.
Works in bull via trend-following breakouts and in bear via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Donchian breakout
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian(20) levels
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (wait for completed 4h bar)
    upper_aligned = align_htf_to_ltf(prices, df_4h, highest_high_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_20)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)  # Volume at least 1.5x average
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 50 for EMA, 20 for volume MA)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Donchian breakout conditions
        price_above_upper = close[i] > upper_aligned[i]
        price_below_lower = close[i] < lower_aligned[i]
        
        # 1d trend filter
        trend_up = close[i] > ema50_1d_aligned[i]
        trend_down = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band AND volume spike AND 1d uptrend AND in session
            if price_above_upper and volume_spike[i] and trend_up and in_session[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower band AND volume spike AND 1d downtrend AND in session
            elif price_below_lower and volume_spike[i] and trend_down and in_session[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price falls below lower band OR 1d trend turns down
            if price_below_lower or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price rises above upper band OR 1d trend turns up
            if price_above_upper or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_4hDonchian20_1dTrend_SessionVolume_v1"
timeframe = "1h"
leverage = 1.0