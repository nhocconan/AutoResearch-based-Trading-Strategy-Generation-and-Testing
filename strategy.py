#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_VolumeTrendFilter
Hypothesis: Breakout above/below weekly Donchian(20) channel with volume confirmation and daily trend filter.
In bull markets: buy breakouts above weekly high with upward daily trend.
In bear markets: sell breakdowns below weekly low with downward daily trend.
Weekly structure filters noise; volume confirms institutional participation.
Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe.
Works in both regimes by following the trend of higher timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate weekly Donchian(20) - highest high and lowest low of past 20 weekly bars
    donchian_high = np.full_like(high_weekly, np.nan)
    donchian_low = np.full_like(low_weekly, np.nan)
    
    lookback = 20
    if len(high_weekly) >= lookback:
        for i in range(lookback, len(high_weekly)):
            donchian_high[i] = np.max(high_weekly[i-lookback:i])
            donchian_low[i] = np.min(low_weekly[i-lookback:i])
    
    # Align weekly Donchian levels to daily timeframe (waits for weekly close)
    donchian_high_daily = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Daily trend filter: EMA(50) slope
    ema_period = 50
    ema = np.full_like(close, np.nan)
    
    if len(close) >= ema_period:
        # Calculate EMA
        alpha = 2.0 / (ema_period + 1)
        ema[0] = close[0]
        for i in range(1, len(close)):
            ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
        
        # EMA slope: positive if current EMA > EMA 5 periods ago
        ema_slope = np.full_like(close, np.nan)
        slope_lookback = 5
        if len(ema) >= slope_lookback:
            for i in range(slope_lookback, len(ema)):
                ema_slope[i] = ema[i] - ema[i - slope_lookback]
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= vol_ma_period:
        for i in range(vol_ma_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_ma_period:i])
    
    volume_ok = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    
    start_idx = max(lookback, ema_period + slope_lookback, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or
            np.isnan(ema_slope[i]) if 'ema_slope' in locals() else True or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long: price breaks above weekly Donchian high + upward daily trend + volume
        if (close[i] > donchian_high_daily[i] and 
            ema_slope[i] > 0 and 
            volume_ok[i]):
            signals[i] = 0.25
        
        # Short: price breaks below weekly Donchian low + downward daily trend + volume
        elif (close[i] < donchian_low_daily[i] and 
              ema_slope[i] < 0 and 
              volume_ok[i]):
            signals[i] = -0.25
        
        # Exit conditions: reverse signal when opposite breakout occurs
        elif close[i] < donchian_low_daily[i] and ema_slope[i] < 0:
            # Flip to short if we were long
            signals[i] = -0.25 if signals[i-1] > 0 else 0.0
        elif close[i] > donchian_high_daily[i] and ema_slope[i] > 0:
            # Flip to long if we were short
            signals[i] = 0.25 if signals[i-1] < 0 else 0.0
        else:
            # Hold current position
            signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals

name = "1d_WeeklyDonchian_Breakout_VolumeTrendFilter"
timeframe = "1d"
leverage = 1.0