#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d ADX(14) for trend strength, 12h Donchian(20) breakout, and volume confirmation.
# Long when 1d ADX > 25, price breaks above 12h Donchian upper band, volume > 1.8x average.
# Short when 1d ADX > 25, price breaks below 12h Donchian lower band, volume > 1.8x average.
# Flat when ADX < 20 (range market). Uses volatility-based sizing and max 10-bar hold.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.
# Works in trending markets (ADX>25) and avoids range-chop (ADX<20).

name = "12h_ADX_12hDonchian_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 12h data for Donchian bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 1d ADX(14) calculation
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha=1/14)
    tr_smooth = np.concatenate([[np.nan], 
                                np.full(13, np.nan), 
                                pd.Series(tr[14:]).ewm(alpha=1/14, adjust=False).mean().values])
    dm_plus_smooth = np.concatenate([[np.nan], 
                                     np.full(13, np.nan), 
                                     pd.Series(dm_plus[14:]).ewm(alpha=1/14, adjust=False).mean().values])
    dm_minus_smooth = np.concatenate([[np.nan], 
                                      np.full(13, np.nan), 
                                      pd.Series(dm_minus[14:]).ewm(alpha=1/14, adjust=False).mean().values])
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth)
    di_minus = 100 * dm_minus_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1e-10, (di_plus + di_minus))
    adx = np.concatenate([[np.nan], 
                          np.full(27, np.nan), 
                          pd.Series(dx[28:]).ewm(alpha=1/14, adjust=False).mean().values])
    
    adx_above_25 = adx > 25
    adx_below_20 = adx < 20
    
    # 12h Donchian(20) bands
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 1d ADX to 12h
    adx_above_25_aligned = align_htf_to_ltf(prices, df_1d, adx_above_25.astype(float))
    adx_below_20_aligned = align_htf_to_ltf(prices, df_1d, adx_below_20.astype(float))
    # Align 12h Donchian bands to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Volatility-based position sizing (ATR-based)
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    vol_factor = np.clip(atr_12h / (close * 0.01), 0.5, 2.0)  # Normalize volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_above_25_aligned[i]) or np.isnan(adx_below_20_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(vol_factor[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only enter when trending (ADX > 25) and not in range (ADX >= 20)
            if adx_above_25_aligned[i] and not adx_below_20_aligned[i]:
                # Long: price breaks above 12h Donchian upper band, volume spike
                if (close[i] > donchian_high_aligned[i] and
                    vol_ratio[i] > 1.8):
                    signals[i] = 0.25 * vol_factor[i]
                    position = 1
                    entry_bar = i
                # Short: price breaks below 12h Donchian lower band, volume spike
                elif (close[i] < donchian_low_aligned[i] and
                      vol_ratio[i] > 1.8):
                    signals[i] = -0.25 * vol_factor[i]
                    position = -1
                    entry_bar = i
        elif position == 1:
            # Long exit: ADX < 20 (range), price breaks below Donchian lower band, or max 10 bars held
            if (adx_below_20_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 10):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 * vol_factor[i]
        elif position == -1:
            # Short exit: ADX < 20 (range), price breaks above Donchian upper band, or max 10 bars held
            if (adx_below_20_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 10):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 * vol_factor[i]
    
    return signals