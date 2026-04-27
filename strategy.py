#!/usr/bin/env python3
"""
1d_WeeklyTrend_DonchianBreakout_30Vol
Hypothesis: Buy breakouts above weekly Donchian high (20) in uptrend, sell breakdowns below weekly Donchian low (20) in downtrend. Volume confirms breakout strength. Weekly trend filter avoids counter-trend trades. Works in bull/bear via trend alignment. Target: 15-25 trades/year on 1d to minimize fee drag.
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
    
    # Get weekly data for trend and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly trend: EMA50
    ema50_weekly = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_weekly_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema50_weekly_aligned[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + uptrend + volume
            if close[i] > upper_band and close[i] > weekly_trend and vol_ok:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly Donchian low + downtrend + volume
            elif close[i] < lower_band and close[i] < weekly_trend and vol_ok:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low or trend turns down
            if close[i] < lower_band or close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high or trend turns up
            if close[i] > upper_band or close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyTrend_DonchianBreakout_30Vol"
timeframe = "1d"
leverage = 1.0