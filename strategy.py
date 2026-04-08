#!/usr/bin/env python3
"""
1h_4h_1d_price_channel_volume_v1
Hypothesis: Use 4h Donchian channel breakouts with volume confirmation and 1d trend filter for direction.
Trades only during active London/New York session (08-20 UTC). Long when price breaks above 4h upper channel with volume and 1d close > 1d EMA50.
Short when price breaks below 4h lower channel with volume and 1d close < 1d EMA50.
Designed to capture breakouts in trending markets while avoiding false breakouts in ranging conditions.
Target: 15-37 trades/year per symbol (60-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_price_channel_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x average of last 24 periods (1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 4h lower Donchian or trend turns down
            if close[i] < donchian_low_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above 4h upper Donchian or trend turns up
            if close[i] > donchian_high_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 4h upper Donchian with volume and 1d trend up
            if (session_filter[i] and 
                close[i] > donchian_high_aligned[i] and 
                vol_confirm[i] and 
                close[i] > ema_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: price breaks below 4h lower Donchian with volume and 1d trend down
            elif (session_filter[i] and 
                  close[i] < donchian_low_aligned[i] and 
                  vol_confirm[i] and 
                  close[i] < ema_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals