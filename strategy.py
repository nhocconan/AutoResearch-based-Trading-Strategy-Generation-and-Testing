#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Donchian(20) breakout with 6h volume confirmation and ADX trend filter
# Long when price breaks above daily Donchian upper band AND 6h ADX > 25 AND volume > 1.3 * avg_volume(20)
# Short when price breaks below daily Donchian lower band AND 6h ADX > 25 AND volume > 1.3 * avg_volume(20)
# Exit when price crosses 6h EMA50 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Daily Donchian provides clear structure with proven breakout edge in both bull/bear markets
# 6h ADX > 25 ensures we only trade in trending conditions, reducing whipsaws
# Volume confirmation filters weak breakouts (reduces false signals)

name = "6h_dailyDonchian20_6hADXTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian(20) channels based on previous daily bar
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    donchian_upper_1d = high_series_1d.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower_1d = low_series_1d.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 6h data ONCE before loop for ADX and EMA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:  # Need sufficient data for ADX and EMA
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h ADX(14) for trend filter
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    plus_dm = np.where((high_6h[1:] - high_6h[:-1]) > (low_6h[:-1] - low_6h[1:]), 
                       np.maximum(high_6h[1:] - high_6h[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.where((low_6h[:-1] - low_6h[1:]) > (high_6h[1:] - high_6h[:-1]), 
                        np.maximum(low_6h[:-1] - low_6h[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First TR is undefined
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h EMA50 for exit signal
    close_series_6h = pd.Series(close_6h)
    ema_50_6h = close_series_6h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily Donchian levels to 6h timeframe (wait for completed daily bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    # Align 6h ADX and EMA to 6h timeframe (no additional delay needed)
    adx_aligned = align_htf_to_ltf(prices, df_6h, adx)
    ema_50_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_50_6h)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(ema_50_6h_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily Donchian upper with ADX > 25 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                adx_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian lower with ADX > 25 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  adx_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 6h EMA50 (trend reversal)
            if close[i] < ema_50_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 6h EMA50 (trend reversal)
            if close[i] > ema_50_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals