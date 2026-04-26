#!/usr/bin/env python3
"""
6h_WeeklyDonchian20_Breakout_1dTrendFilter_VolumeSpike_v1
Hypothesis: 6h Donchian(20) breakout strategy filtered by 1d trend and weekly pivot direction.
- Uses 6h timeframe for lower trade frequency (target: 50-150 total trades over 4 years)
- Donchian breakout with 20-period lookback on 6h candles
- Long when price breaks above 6h Donchian high AND 1d uptrend AND weekly bullish bias
- Short when price breaks below 6h Donchian low AND 1d downtrend AND weekly bearish bias
- Volume confirmation required (2x 20-period average)
- Weekly trend from 1w EMA50 slope (bullish if rising, bearish if falling)
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by requiring alignment across 6h, 1d, and 1w timeframes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d EMA50 slope for trend direction (rising/falling)
    ema50_1d_series = pd.Series(ema50_1d)
    ema50_slope = ema50_1d_series.diff(periods=3).values  # 3-bar slope
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope)
    
    # Load 1w data ONCE before loop for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for weekly trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1w EMA50 slope for weekly trend direction
    ema50_1w_series = pd.Series(ema50_1w)
    ema50_1w_slope = ema50_1w_series.diff(periods=3).values  # 3-bar slope
    ema50_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_slope)
    
    # Calculate Donchian channels on 6h (20-period lookback)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian/volume, 50 for EMAs)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_1w_slope_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions
        price_above_dchigh = close[i] > donchian_high[i]
        price_below_dclow = close[i] < donchian_low[i]
        
        # 1d trend conditions
        trend_1d_up = close[i] > ema50_1d_aligned[i]
        trend_1d_down = close[i] < ema50_1d_aligned[i]
        slope_1d_up = ema50_slope_aligned[i] > 0
        slope_1d_down = ema50_slope_aligned[i] < 0
        
        # Weekly trend conditions
        slope_1w_up = ema50_1w_slope_aligned[i] > 0
        slope_1w_down = ema50_1w_slope_aligned[i] < 0
        
        if position == 0:
            # Long: price breaks above Donchian high AND 1d uptrend AND weekly bullish bias
            if (price_above_dchigh and trend_1d_up and slope_1d_up and 
                slope_1w_up and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND 1d downtrend AND weekly bearish bias
            elif (price_below_dclow and trend_1d_down and slope_1d_down and 
                  slope_1w_down and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below Donchian low OR 1d trend turns down OR weekly turns bearish
            if (price_below_dclow or not trend_1d_up or not slope_1d_up or not slope_1w_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high OR 1d trend turns up OR weekly turns bullish
            if (price_above_dchigh or not trend_1d_down or not slope_1d_down or not slope_1w_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_1dTrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0