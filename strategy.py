#!/usr/bin/env python3
"""
1d_PriceChannel_Breakout_WeeklyTrend_1dVolume
Hypothesis: Trade weekly trend direction with daily price channel breakouts.
Long when weekly EMA21 is rising and price breaks above daily Donchian(20) high with volume > 1.5x average.
Short when weekly EMA21 is falling and price breaks below daily Donchian(20) low with volume > 1.5x average.
Exit when price crosses the weekly EMA21 or after 5 days. Designed for 1d timeframe to capture trends
while avoiding whipsaw in sideways markets. Weekly trend filter reduces false breakouts, volume confirms
institutional interest. Targets 10-25 trades/year by requiring both trend alignment and breakout.
Works in bull markets by following uptrend breakouts, in bear by shorting downtrend breakdowns.
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
    
    # Get weekly data for EMA21 trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Weekly EMA21
    ema_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 21:
        ema_weekly[20] = np.mean(close_weekly[:21])
        for i in range(21, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2/22) + (ema_weekly[i-1] * 20/22)
    
    # Align weekly EMA to daily
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Daily volume average (20-period)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_weekly_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        bars_since_entry += 1
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Weekly uptrend + Donchian breakout up + volume
            if (i > 0 and not np.isnan(ema_weekly_aligned[i-1]) and 
                ema_weekly_aligned[i] > ema_weekly_aligned[i-1] and  # Rising weekly EMA
                close[i] > donchian_high[i] and vol_confirm):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Weekly downtrend + Donchian breakout down + volume
            elif (i > 0 and not np.isnan(ema_weekly_aligned[i-1]) and 
                  ema_weekly_aligned[i] < ema_weekly_aligned[i-1] and  # Falling weekly EMA
                  close[i] < donchian_low[i] and vol_confirm):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Long exit: Weekly trend turns down OR price crosses below weekly EMA OR max 5 days
            weekly_trend_down = (i > 0 and not np.isnan(ema_weekly_aligned[i-1]) and 
                                ema_weekly_aligned[i] < ema_weekly_aligned[i-1])
            price_below_ema = close[i] < ema_weekly_aligned[i]
            max_days_reached = bars_since_entry >= 5
            
            if weekly_trend_down or price_below_ema or max_days_reached:
                signals[i] = -0.25  # reverse to short
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Weekly trend turns up OR price crosses above weekly EMA OR max 5 days
            weekly_trend_up = (i > 0 and not np.isnan(ema_weekly_aligned[i-1]) and 
                              ema_weekly_aligned[i] > ema_weekly_aligned[i-1])
            price_above_ema = close[i] > ema_weekly_aligned[i]
            max_days_reached = bars_since_entry >= 5
            
            if weekly_trend_up or price_above_ema or max_days_reached:
                signals[i] = 0.25  # reverse to long
                position = 1
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_PriceChannel_Breakout_WeeklyTrend_1dVolume"
timeframe = "1d"
leverage = 1.0