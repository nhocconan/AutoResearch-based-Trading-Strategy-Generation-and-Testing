#!/usr/bin/env python3
"""
12h_1w_DonchianBreakout_TrendFilter_V1
Hypothesis: Weekly Donchian(20) breakouts with 1w trend filter (EMA34) and volume confirmation capture strong moves in both bull and bear markets. The 12h timeframe provides timely entry while the weekly filter reduces whipsaw. Volume surge confirms breakout authenticity. Target: 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for Donchian and EMA
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly Donchian channels (20-period)
    donchian_high = np.full_like(close_weekly, np.nan)
    donchian_low = np.full_like(close_weekly, np.nan)
    for i in range(20, len(close_weekly)):
        donchian_high[i] = np.max(high_weekly[i-20:i])
        donchian_low[i] = np.min(low_weekly[i-20:i])
    
    # Weekly EMA34 for trend filter
    close_series = pd.Series(close_weekly)
    ema34_weekly = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # 12h data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 24-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 24:
            volume_avg[i] = np.mean(volume[i-24:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (2.0 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if NaN in critical values
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema34_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        trend = ema34_weekly_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above upper Donchian with uptrend and volume
            if price > upper and price > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with downtrend and volume
            elif price < lower and price < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below weekly EMA34 or Donchian low
            if price < trend or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above weekly EMA34 or Donchian high
            if price > trend or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_DonchianBreakout_TrendFilter_V1"
timeframe = "12h"
leverage = 1.0