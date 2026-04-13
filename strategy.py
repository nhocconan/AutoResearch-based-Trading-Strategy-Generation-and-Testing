#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
    # Long when price breaks above 20-day high AND weekly close > weekly EMA200 AND 1d volume > 1.5x 20-day volume MA.
    # Short when price breaks below 20-day low AND weekly close < weekly EMA200 AND 1d volume > 1.5x 20-day volume MA.
    # Exit when price re-enters the 20-day Donchian channel (mean reversion).
    # Uses Donchian for structure, weekly EMA for trend, volume for confirmation.
    # Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian calculation and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels from previous day
    # Upper = max(high_1d[-20:-1])
    # Lower = min(low_1d[-20:-1])
    high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma_20  # Already shifted by rolling window
    donchian_lower = low_ma_20   # Already shifted by rolling window
    
    # Calculate 20-day volume MA for confirmation
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're on 1d)
    donchian_upper_aligned = donchian_upper  # Already on 1d
    donchian_lower_aligned = donchian_lower  # Already on 1d
    vol_ma_20_aligned = vol_ma_20            # Already on 1d
    
    # Align weekly EMA200 to 1d timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_spike = volume_1d[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Price relative to Donchian levels
        price_above_upper = close_1d[i] > donchian_upper_aligned[i]
        price_below_lower = close_1d[i] < donchian_lower_aligned[i]
        price_in_channel = (close_1d[i] >= donchian_lower_aligned[i]) & (close_1d[i] <= donchian_upper_aligned[i])
        
        # Trend filter: weekly close vs EMA200
        trend_bullish = close_1d[i] > ema200_1w_aligned[i]  # Using 1d close vs weekly EMA200
        trend_bearish = close_1d[i] < ema200_1w_aligned[i]
        
        # Entry conditions
        if price_above_upper and trend_bullish and volume_spike and position != 1:
            position = 1
            signals[i] = position_size
        elif price_below_lower and trend_bearish and volume_spike and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions: price re-enters Donchian channel
        elif price_in_channel and position != 0:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_breakout_ema_volume_v1"
timeframe = "1d"
leverage = 1.0