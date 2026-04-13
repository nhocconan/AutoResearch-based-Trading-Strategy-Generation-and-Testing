#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h Donchian breakout with 1d trend filter (EMA50) and volume confirmation.
    # 4h Donchian provides directional structure, 1h EMA50 confirms short-term trend alignment.
    # Volume filter ensures breakout has participation. Session filter (08-20 UTC) reduces noise.
    # Target: 60-150 total trades over 4 years = 15-37/year for 1h.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_window = 20
    donchian_high_4h = pd.Series(high_4h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume MA20 for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1d volume MA20
        volume_filter = volume[i] > volume_ma_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > donchian_high_aligned[i]  # Break above upper channel
        breakout_short = close[i] < donchian_low_aligned[i]  # Break below lower channel
        
        # Trend filter: price above/below 1d EMA50
        trend_long = close[i] > ema50_aligned[i]
        trend_short = close[i] < ema50_aligned[i]
        
        # Entry conditions: breakout in direction of trend with volume confirmation
        long_entry = breakout_long and trend_long and volume_filter
        short_entry = breakout_short and trend_short and volume_filter
        
        # Exit conditions: price returns to opposite Donchian level
        long_exit = close[i] < donchian_low_aligned[i]
        short_exit = close[i] > donchian_high_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_donchian_breakout_ema50_volume_v1"
timeframe = "1h"
leverage = 1.0