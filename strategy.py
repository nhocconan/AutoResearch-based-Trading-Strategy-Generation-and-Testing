#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d trend filter.
- Direction from 4h: long when price above 4h Donchian upper (20), short when below lower (20)
- Entry timing on 1h: only enter on break of 1h hourly high/low in direction of 4h trend
- Volume filter: require 1h volume > 1.5x 20-period volume MA
- Trend filter: 1d EMA50 - only long when price above, short when below
- Exit when 4h Donchian direction reverses or volume drops below 0.5x average
- Position size 0.20 to limit risk
- Uses multi-timeframe alignment: 4h for structure/trend, 1h for precise entry
- Designed for 15-30 trades/year to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h trend direction (1=long bias, -1=short bias, 0=neutral)
    trend_4h = np.zeros(len(prices))
    for i in range(len(prices)):
        if donchian_high[i] == donchian_high[i]:  # not nan
            if prices['close'].iloc[i] > donchian_high[i]:
                trend_4h[i] = 1
            elif prices['close'].iloc[i] < donchian_low[i]:
                trend_4h[i] = -1
            else:
                trend_4h[i] = trend_4h[i-1] if i > 0 else 0
        else:
            trend_4h[i] = trend_4h[i-1] if i > 0 else 0
    
    # Align 4h trend to 1h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h hourly high/low for entry timing
    hourly_high = pd.Series(high).rolling(window=2, min_periods=2).max().values  # previous hour high
    hourly_low = pd.Series(low).rolling(window=2, min_periods=2).min().values   # previous hour low
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(hourly_high[i]) or np.isnan(hourly_low[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        trend = trend_4h_aligned[i]
        ema_filter = ema_50_1d_aligned[i]
        hh = hourly_high[i]
        hl = hourly_low[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0  # close position outside session
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for entries aligned with 4h trend and 1d filter
            # Long: 4h trend up, price above 1d EMA50, break hourly high, volume spike
            if (trend == 1 and price > ema_filter and 
                price > hh and vol > 1.5 * vol_ma):
                signals[i] = 0.20
                position = 1
            # Short: 4h trend down, price below 1d EMA50, break hourly low, volume spike
            elif (trend == -1 and price < ema_filter and 
                  price < hl and vol > 1.5 * vol_ma):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Hold long while: 4h trend up, price above 1d EMA50, volume adequate
            # Exit if trend reverses, price breaks below EMA50, or volume drops
            if (trend != 1 or price < ema_filter or vol < 0.5 * vol_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Hold short while: 4h trend down, price below 1d EMA50, volume adequate
            # Exit if trend reverses, price breaks above EMA50, or volume drops
            if (trend != -1 or price > ema_filter or vol < 0.5 * vol_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian_1dEMA50_VolumeFilter"
timeframe = "1h"
leverage = 1.0