#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h 4h/1d Confluence with Volume and Session Filter
# Hypothesis: Price breaking 4h Donchian channel with 1d EMA trend and volume
# confirmation during active London/NY session (08-20 UTC) captures
# institutional moves in both bull and bear markets. 4h provides directional bias,
# 1d EMA filters counter-trend moves, volume confirms institutional participation.
# Target: 15-30 trades/year (60-120 over 4 years).

name = "1h_4h1d_donchian_volume_session_v1"
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
    
    # Session filter: 08-20 UTC (London/NY overlap)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channel (20-period high/low)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (previous 20 periods)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4-period lookback for 4h Donchian (each 4h bar = 4 1h bars)
    lookback_4h = 20
    donchian_high = pd.Series(high_4h).rolling(window=lookback_4h, min_periods=lookback_4h).max().values
    donchian_low = pd.Series(low_4h).rolling(window=lookback_4h, min_periods=lookback_4h).min().values
    
    # Align to 1h timeframe (use previous 4h bar's levels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for trend filter (50 EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.8x 24-period average (institutional participation)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or not session_mask[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend turns bearish or volume drops
            if (low[i] < donchian_low_aligned[i] or close[i] < ema_50_1d_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend turns bullish or volume drops
            if (high[i] > donchian_high_aligned[i] or close[i] > ema_50_1d_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above 4h Donchian high with 1d uptrend and volume
            if (high[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short: price breaks below 4h Donchian low with 1d downtrend and volume
            elif (low[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals