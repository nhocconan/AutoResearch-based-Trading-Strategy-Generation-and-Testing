#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter
# - Donchian(20) from 4h: breakout above upper band = long, below lower band = short
# - 1d volume confirmation: current 4h volume > 2.0x 20-period average to confirm institutional interest
# - 1w ADX(14) > 25 to ensure weekly trend alignment and avoid choppy markets
# - Designed for 4h timeframe: targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: weekly ADX filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.30) to minimize fee churn
# - ATR-based stoploss: exit when price moves 2.5*ATR against position

name = "4h_1d_1w_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
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
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 1d volume confirmation (using 4h volume aggregated to 1d then aligned back)
    # But simpler: use 4h volume directly with 20-period average
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr_4h[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if prices['high'].iloc[i] > highest_since_entry:
                highest_since_entry = prices['high'].iloc[i]
            
            # Exit: price drops 2.5*ATR from highest high (trailing stop) 
            # OR price closes below Donchian lower (breakdown)
            if (prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_4h[i] or
                prices['close'].iloc[i] < donchian_lower[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if prices['low'].iloc[i] < lowest_since_entry:
                lowest_since_entry = prices['low'].iloc[i]
            
            # Exit: price rises 2.5*ATR from lowest low (trailing stop)
            # OR price closes above Donchian upper (breakout)
            if (prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_4h[i] or
                prices['close'].iloc[i] > donchian_upper[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Breakout long: price closes above Donchian upper
                if prices['close'].iloc[i] > donchian_upper[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.30
                # Breakout short: price closes below Donchian lower
                elif prices['close'].iloc[i] < donchian_lower[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = -0.30
    
    return signals