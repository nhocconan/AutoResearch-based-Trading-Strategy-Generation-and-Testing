#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian Breakout with 12h EMA Trend Filter and Volume Spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Donchian(20) upper band AND price > 12h EMA50 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Donchian(20) lower band AND price < 12h EMA50 AND volume > 2.0 * 4h volume MA(20).
- Exit: Long exits when price crosses below Donchian(10) middle band; Short exits when price crosses above Donchian(10) middle band.
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian breakouts capture strong momentum; EMA50 ensures trend alignment; volume spike confirms institutional participation.
- Works in bull (breakouts in uptrend) and bear (breakdowns in downtrend).
- Avoids false breakouts via strict volume and trend filters.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA50 for 12h
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Donchian channels on 4h data
    donchian_period = 20
    donchian_exit_period = 10
    
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    highest_high_exit = pd.Series(high).rolling(window=donchian_exit_period, min_periods=donchian_exit_period).max().values
    lowest_low_exit = pd.Series(low).rolling(window=donchian_exit_period, min_periods=donchian_exit_period).min().values
    
    # Calculate 4h volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, donchian_period, 20)  # EMA50 needs 50, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(highest_high_exit[i]) or 
            np.isnan(lowest_low_exit[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above Donchian(20) upper band AND price > 12h EMA50 (uptrend)
                if curr_high > highest_high[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian(20) lower band AND price < 12h EMA50 (downtrend)
                elif curr_low < lowest_low[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below Donchian(10) middle band
            mid_exit = (highest_high_exit[i] + lowest_low_exit[i]) / 2
            if curr_close < mid_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Donchian(10) middle band
            mid_exit = (highest_high_exit[i] + lowest_low_exit[i]) / 2
            if curr_close > mid_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0