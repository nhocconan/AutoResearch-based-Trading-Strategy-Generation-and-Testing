#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and ATR stoploss
# - Primary timeframe: 12h for lower trade frequency (target 12-37 trades/year)
# - Entry: Price breaks above/below 20-period Donchian channel with volume spike
# - HTF filter: 1d ADX > 25 to ensure daily trend alignment
# - Exit: ATR-based trailing stop (3*ATR) or opposite Donchian break
# - Position sizing: 0.25 discrete to minimize fee churn
# - Works in bull/bear markets: ADX filter avoids chop, Donchian captures breakouts

name = "12h_1d_donchian_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14) for trend filter
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
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
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
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 12h ATR(14) for stoploss
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_12h[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        close_price = prices['close'].iloc[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
            # Exit: ATR trailing stop or price breaks below Donchian low
            if close_price < highest_since_entry - 3.0 * atr_12h[i] or close_price < donchian_low[i]:
                position = 0
                highest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
            # Exit: ATR trailing stop or price breaks above Donchian high
            if close_price > lowest_since_entry + 3.0 * atr_12h[i] or close_price > donchian_high[i]:
                position = 0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Breakout long: price closes above Donchian high
                if close_price > donchian_high[i]:
                    position = 1
                    entry_price = close_price
                    highest_since_entry = prices['high'].iloc[i]
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = 0.25
                # Breakout short: price closes below Donchian low
                elif close_price < donchian_low[i]:
                    position = -1
                    entry_price = close_price
                    highest_since_entry = prices['high'].iloc[i]
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = -0.25
    
    return signals