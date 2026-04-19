#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 3-bar Donchian breakout with 1-day volume confirmation and ADX trend filter.
# Long when: Close breaks above Donchian high (3-bar max), 1d volume > 1.5x 20-period avg, ADX > 20
# Short when: Close breaks below Donchian low (3-bar min), 1d volume > 1.5x 20-period avg, ADX > 20
# Exit when: Price returns to opposite Donchian level (long exits at Donchian low, short at Donchian high)
# Uses actual price breakouts for momentum, volume confirms institutional interest, ADX ensures trending market.
# Target: 25-40 trades/year per symbol. Works in bull (breakouts up) and bear (breakdowns down).
name = "4h_Donchian3_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for volume confirmation and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 3-bar Donchian channels (highest high/lowest low of last 3 bars)
    donch_high = pd.Series(high).rolling(window=3, min_periods=3).max().values
    donch_low = pd.Series(low).rolling(window=3, min_periods=3).min().values
    
    # 1-day volume average for confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # ADX calculation on 1-day data (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = wilders_smoothing(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume_1d[i // 96] if i >= 96 else volume_1d[0]  # Approximate 1d volume index (96*15m=24h, but we use 1d data aligned)
        vol_ma = vol_ma_20_1d_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above 3-bar Donchian high, volume spike, ADX > 20
            if (price > donch_high[i] and vol > 1.5 * vol_ma and adx_val > 20):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below 3-bar Donchian low, volume spike, ADX > 20
            elif (price < donch_low[i] and vol > 1.5 * vol_ma and adx_val > 20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below 3-bar Donchian low
            if price <= donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above 3-bar Donchian high
            if price >= donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals