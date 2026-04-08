#!/usr/bin/env python3

name = "1d_donchian_20_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (call ONCE before loop)
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA(20) for trend direction
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 40  # Need weekly EMA and Donchian buffers
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_weekly[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned weekly EMA for current daily bar
        ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)[i]
        
        # Trend filter: price above/below weekly EMA20
        price_above_weekly_ema = close[i] > ema_20_weekly_aligned
        price_below_weekly_ema = close[i] < ema_20_weekly_aligned
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below daily Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above daily Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_confirm:
                # Long entry: price breaks above Donchian high AND above weekly EMA20
                if close[i] > donchian_high[i] and price_above_weekly_ema:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low AND below weekly EMA20
                elif close[i] < donchian_low[i] and price_below_weekly_ema:
                    position = -1
                    signals[i] = -0.25
    
    return signals