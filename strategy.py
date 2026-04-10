#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h volume spike + 1d ADX regime filter
# - Primary signal: Price breaks above/below 20-period Donchian channel on 6h
# - Volume confirmation: 12h volume > 1.3x 20-period average volume (avoid low-participation breakouts)
# - Regime filter: 1d ADX > 25 (trending market) enables breakout continuation; ADX < 20 enables mean reversion at Donchian bands
# - Works in bull/bear: In strong trends (ADX > 25), breakouts continue; in weak trends/ranges (ADX < 20), fade Donchian touches
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(20)

name = "6h_12h_1d_donchian_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h volume spike filter
    volume_12h = df_12h['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.3 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    # Pre-compute 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(atr_14 != 0, atr_14, 1e-10)
    minus_di = 100 * minus_dm_smooth / np.where(atr_14 != 0, atr_14, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h Donchian Channel (20)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 6h ATR(20) for stoploss
    tr_6h1 = high_6h - low_6h
    tr_6h2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr_6h3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h[0] = tr_6h1[0]
    atr_20 = pd.Series(tr_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_6h[i] < donchian_mid[i] or close_6h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_6h[i] > donchian_mid[i] or close_6h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume spike and ADX regime filter
            # In weak trends/ranges (ADX < 20): fade Donchian touches (mean reversion)
            # In strong trends (ADX > 25): breakout continuation
            if volume_spike_aligned[i]:
                if adx_aligned[i] < 20:  # weak trend/ranging market - mean reversion
                    # Long: price touches lower Donchian band
                    if close_6h[i] <= donchian_low[i] * 1.001:  # small buffer for noise
                        position = 1
                        entry_price = close_6h[i]
                        signals[i] = 0.25
                    # Short: price touches upper Donchian band
                    elif close_6h[i] >= donchian_high[i] * 0.999:
                        position = -1
                        entry_price = close_6h[i]
                        signals[i] = -0.25
                elif adx_aligned[i] > 25:  # strong trending market - breakout continuation
                    # Long: price breaks above upper Donchian band
                    if close_6h[i] > donchian_high[i]:
                        position = 1
                        entry_price = close_6h[i]
                        signals[i] = 0.25
                    # Short: price breaks below lower Donchian band
                    elif close_6h[i] < donchian_low[i]:
                        position = -1
                        entry_price = close_6h[i]
                        signals[i] = -0.25
    
    return signals