#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when close breaks above Donchian upper (20) AND price > 1w EMA50 AND volume > 1.5 * 1d volume MA(20);
         Short when close breaks below Donchian lower (20) AND price < 1w EMA50 AND volume > 1.5 * 1d volume MA(20).
- Exit: Long exits when close crosses below Donchian lower; Short exits when close crosses above Donchian upper.
- Signal size: 0.25 discrete to control fee drag.
- Uses Donchian channels for structure, volume confirmation for participation, and EMA50 trend filter to avoid counter-trend trades.
- Designed for both bull (breakouts with trend) and bear (short breakdowns with trend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA50 for 1w trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) from 1d data
    # Donchian Upper = highest high over 20 periods
    # Donchian Lower = lowest low over 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold for balanced entry
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above Donchian upper AND price > 1w EMA50 (uptrend)
                if curr_close > donchian_upper[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close breaks below Donchian lower AND price < 1w EMA50 (downtrend)
                elif curr_close < donchian_lower[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when close crosses below Donchian lower
            if curr_close < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when close crosses above Donchian upper
            if curr_close > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0