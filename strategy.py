#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with daily Donchian(20) breakout + volume confirmation + daily ADX(14) trend filter.
# Daily Donchian(20) identifies 20-day price extremes. Breakout captures momentum.
# Volume surge confirms institutional participation in the breakout.
# Daily ADX > 25 ensures trades occur in trending markets, avoiding chop.
# Works in bull/bear by following strong trends from price extremes.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Daily Donchian Channel (20) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian bands
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # === Daily ADX (14) for trend strength ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close, 1))  # Note: using close from prices for continuity
    tr3 = np.abs(low_1d - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(atr_14 != 0, 100 * plus_dm_smooth / atr_14, 0)
    minus_di = np.where(atr_14 != 0, 100 * minus_dm_smooth / atr_14, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_14 = wilders_smoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # === Daily volume for surge confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # Volume surge: current 1d volume > 1.3x 20-period average
        vol_1d_current = volume_1d[i]  # Already daily volume
        vol_surge = vol_1d_current > vol_ma_20_aligned[i] * 1.3
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25.0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above daily Donchian upper + volume surge + trending
            if price > donchian_upper_aligned[i] and vol_surge and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below daily Donchian lower + volume surge + trending
            elif price < donchian_lower_aligned[i] and vol_surge and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite breakout
        elif position == 1:
            # Exit long if price breaks below daily Donchian lower
            if price < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above daily Donchian upper
            if price > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolume1.3x_1dADX25_TrendFilter"
timeframe = "12h"
leverage = 1.0