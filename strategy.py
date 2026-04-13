#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
    # Long when price breaks above Donchian high(20) AND 1d close > 1d EMA50 (bullish trend) AND 4h volume > 1.5x 20-period MA.
    # Short when price breaks below Donchian low(20) AND 1d close < 1d EMA50 (bearish trend) AND 4h volume > 1.5x 20-period MA.
    # Exit when price re-enters Donchian channel (mean reversion).
    # Uses Donchian for structure, 1d EMA for trend, volume for confirmation.
    # Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 4h data for Donchian channels and volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume MA(20)
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 4h timeframe (no alignment needed as we're already on 4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma_4h_aligned[i]
        
        # Price relative to Donchian channels
        price_above_high = close[i] > donchian_high_aligned[i]
        price_below_low = close[i] < donchian_low_aligned[i]
        price_in_channel = (close[i] >= donchian_low_aligned[i]) & (close[i] <= donchian_high_aligned[i])
        
        # Trend filter: 1d close vs EMA50
        trend_bullish = close[i] > ema50_1d_aligned[i]  # Using 4h close vs 1d EMA50
        trend_bearish = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        if price_above_high and trend_bullish and volume_spike and position != 1:
            position = 1
            signals[i] = position_size
        elif price_below_low and trend_bearish and volume_spike and position != -1:
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

name = "4h_1d_donchian_breakout_ema_volume_v1"
timeframe = "4h"
leverage = 1.0