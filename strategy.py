#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and 1d volume spike confirmation.
# Long when price breaks above 6h Donchian upper band AND 1d ADX > 25 AND 1d volume > 1.5x 20-period median.
# Short when price breaks below 6h Donchian lower band AND 1d ADX > 25 AND 1d volume > 1.5x 20-period median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian provides clear breakout structure, 1d ADX filters for trending markets (works in bull/bear),
# 1d volume spike confirms institutional participation. Target: 12-25 trades/year on 6h (50-100 total over 4 years).

name = "6h_Donchian20_Breakout_1dADX_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h Donchian bands (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ADX (14-period) for trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_first_1d = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr_1d = np.concatenate([[tr_first_1d], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_first = np.maximum(high_1d[0] - high_1d[0], 0)  # Always 0 for first bar
    dm_minus_first = np.maximum(low_1d[0] - low_1d[0], 0)   # Always 0 for first bar
    dm_plus = np.concatenate([[dm_plus_first], dm_plus])
    dm_minus = np.concatenate([[dm_minus_first], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d volume median (20-period for stability)
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).median().values
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, ADX, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_median_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current 6h volume > 1.5x 1d volume median (scaled to 6h)
        # Approximate: 1d volume median / 4 (since 4x 6h bars in 1d) then compare
        vol_median_6h_approx = vol_median_1d_aligned[i] / 4.0
        if vol_median_6h_approx <= 0 or np.isnan(vol_median_6h_approx):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_6h_approx * 1.5)
        
        # Trend filter: 1d ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian upper AND strong trend AND volume confirmation
            if (curr_high > donchian_upper[i] and 
                strong_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Break below Donchian lower AND strong trend AND volume confirmation
            elif (curr_low < donchian_lower[i] and 
                  strong_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower OR trend weakens (ADX < 20)
            elif (curr_low < donchian_lower[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper OR trend weakens (ADX < 20)
            elif (curr_high > donchian_upper[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals