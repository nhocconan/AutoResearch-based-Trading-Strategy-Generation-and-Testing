#!/usr/bin/env python3
"""
1d_donchian_1w_trend_volume_v2
Hypothesis: On daily timeframe, break above/below Donchian channel (20-period) with weekly trend confirmation and volume spike. Uses weekly EMA50 for trend filter and volume > 2x 20-period average for confirmation. Designed to work in both bull (breakouts) and bear (breakdowns) markets with minimal trades to avoid fee drag. Targets 10-20 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_1w_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    w_close = df_1w['close'].values
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA50 for trend filter
    weekly_close_series = pd.Series(w_close)
    ema50 = weekly_close_series.ewm(span=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Volume filter: daily volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if weekly EMA not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Donchian breakout/breakdown
        breakout = close[i] > donchian_high[i-1]  # Above prior period high
        breakdown = close[i] < donchian_low[i-1]  # Below prior period low
        
        # Weekly trend filter: price above/below EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 2.0
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price breaks below Donchian low
            if breakdown:
                exit_long = True
            # Exit when trend turns down
            elif not uptrend:
                exit_long = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 1.0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price breaks above Donchian high
            if breakout:
                exit_short = True
            # Exit when trend turns up
            elif not downtrend:
                exit_short = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 1.0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high AND uptrend AND volume confirmed
            long_entry = breakout and uptrend and vol_confirmed
            
            # Short entry: breakdown below Donchian low AND downtrend AND volume confirmed
            short_entry = breakdown and downtrend and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals