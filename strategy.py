#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w ADX(14) > 25 trend filter
# - Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average and weekly ADX > 25
# - Short when price breaks below Donchian(20) low with volume > 1.5x 20-period average and weekly ADX > 25
# - Exit on opposite Donchian(10) break or ATR(14) stoploss (2.0x ATR)
# - Designed for 12h timeframe: targets 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Weekly ADX filter ensures we only trade with strong higher timeframe trend, reducing whipsaw in ranging markets
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "12h_1d_1w_donchian_volume_adx_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 12h Donchian channels
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Donchian(20) for entry
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for exit
    highest_high_10 = pd.Series(high_12h).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low_12h).rolling(window=10, min_periods=10).min().values
    
    # Pre-compute 12h ATR(14) for stoploss
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    
    atr_14 = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i]) or np.isnan(highest_high_10[i]) or np.isnan(lowest_low_10[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss, Donchian(10) breakdown, or loss of weekly trend
            if (prices['close'].iloc[i] < entry_price - 2.0 * atr_14[i] or 
                prices['close'].iloc[i] < lowest_low_10[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss, Donchian(10) breakout, or loss of weekly trend
            if (prices['close'].iloc[i] > entry_price + 2.0 * atr_14[i] or 
                prices['close'].iloc[i] > highest_high_10[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian(20) breakout with volume spike and weekly trend filter
            if vol_spike[i] and adx_aligned[i] > 25:
                # Long signal: price breaks above Donchian(20) high
                if prices['close'].iloc[i] > highest_high_20[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian(20) low
                elif prices['close'].iloc[i] < lowest_low_20[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals