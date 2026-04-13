#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volume confirmation.
    # Long when price breaks above Donchian upper AND 1d EMA50 rising AND 4h volume > 1.5x ATR-scaled average.
    # Short when price breaks below Donchian lower AND 1d EMA50 falling AND 4h volume > 1.5x ATR-scaled average.
    # Exit when price retouches Donchian midpoint.
    # Uses discrete sizing (0.25) and volume-ATR confirmation to target 75-200 trades over 4 years.
    # Works in bull/bear via EMA50 trend filter and volume-ATR confirmation reducing false breakouts.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 4h data for Donchian channels and volume/ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate ATR(14) for volume scaling
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume average and ATR-scaled threshold
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * (1.0 + atr / close_4h)  # ATR-scaled volume threshold
    
    # Align all indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_threshold_aligned = align_htf_to_ltf(prices, df_4h, vol_threshold)
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_threshold_aligned[i]) or np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > ATR-scaled average
        volume_confirm = volume_4h_aligned[i] > vol_threshold_aligned[i]
        
        # Price relative to Donchian channels
        price_above_upper = close_4h_aligned[i] > donchian_high_aligned[i]
        price_below_lower = close_4h_aligned[i] < donchian_low_aligned[i]
        price_at_mid = np.abs(close_4h_aligned[i] - donchian_mid_aligned[i]) < (donchian_high_aligned[i] - donchian_low_aligned[i]) * 0.05
        
        # Trend filter: 1d EMA50 direction (using slope)
        ema50_slope = ema50_1d_aligned[i] - ema50_1d_aligned[i-1] if i > 0 else 0
        trend_bullish = ema50_slope > 0
        trend_bearish = ema50_slope < 0
        
        # Entry conditions
        if price_above_upper and trend_bullish and volume_confirm and position != 1:
            position = 1
            signals[i] = position_size
        elif price_below_lower and trend_bearish and volume_confirm and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions: price retouches Donchian midpoint
        elif price_at_mid and position != 0:
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

name = "4h_1d_donchian_breakout_ema50_volume_atr_v1"
timeframe = "4h"
leverage = 1.0