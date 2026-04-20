# 1h_4h1d_DonchianBreakout_TrendFilter_V1
# 1h timeframe strategy using 4h Donchian breakouts with 1d trend filter
# Enters long when 1h price breaks above 4h upper Donchian band and 1d trend is up
# Enters short when 1h price breaks below 4h lower Donchian band and 1d trend is down
# Uses 1d trend (EMA50) to filter direction and avoid counter-trend trades
# Session filter (08-20 UTC) to reduce noise
# Fixed size 0.20 to manage risk
# Target: 15-37 trades/year (60-150 total over 4 years)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20) for breakout signals
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        trend_val = ema_50_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(trend_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h upper Donchian band with up 1d trend
            if close_val > upper_val and close_val > trend_val:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h lower Donchian band with down 1d trend
            elif close_val < lower_val and close_val < trend_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 4h lower Donchian band
            if close_val < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above 4h upper Donchian band
            if close_val > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_DonchianBreakout_TrendFilter_V1"
timeframe = "1h"
leverage = 1.0