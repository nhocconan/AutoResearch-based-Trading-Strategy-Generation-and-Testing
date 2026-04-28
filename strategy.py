#!/usr/bin/env python3
"""
12h_PriceChannel_Breakout_VolumeTrend
Hypothesis: Uses 12-hour price channels (20-period high/low) with volume confirmation and 1-day trend filter.
Trades breakouts above channel high in uptrend and breakdowns below channel low in downtrend.
Volume spike filters false breakouts. Designed for trending markets with controlled frequency.
Targets 12-30 trades per year to minimize fee drag while capturing significant moves.
Works in both bull and bear markets by following the 1-day trend direction.
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12-hour price channel (20-period high/low)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (>2.0x 20-period MA for strict filtering)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1-day EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Price relative to 12-hour channel
        price_above_channel = close[i] > high_max_20[i-1]
        price_below_channel = close[i] < low_min_20[i-1]
        
        # Entry logic:
        # Long: Breakout above channel high in uptrend with volume
        long_entry = vol_confirm and trend_up and price_above_channel
        
        # Short: Breakdown below channel low in downtrend with volume
        short_entry = vol_confirm and trend_down and price_below_channel
        
        # Exit logic: Opposite channel touch or trend reversal
        long_exit = (close[i] < low_min_20[i-1]) or not trend_up
        short_exit = (close[i] > high_max_20[i-1]) or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_PriceChannel_Breakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0