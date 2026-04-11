#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12-hour Donchian breakout + volume confirmation + ADX trend filter.
# Uses 12h Donchian channels to capture breakouts, confirmed by volume surge and ADX>25 for trend strength.
# Designed for low trade frequency (~20-40/year) to minimize fee decay while capturing strong trends.
# Works in bull/bear markets by filtering breakouts with ADX and volume to avoid false signals.

name = "4h_12h_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period high/low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h ADX for trend strength
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close, 1))
    tr3 = np.abs(low_12h - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First value has no previous close
    
    plus_dm = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di = np.where(tr14 > 0, plus_dm14 / tr14 * 100, 0)
    minus_di = np.where(tr14 > 0, minus_dm14 / tr14 * 100, 0)
    dx = np.where((plus_di + minus_di) > 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate volume confirmation (current volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 50 to ensure all indicators are valid
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: Donchian breakout with volume surge and ADX>25
        long_breakout = high[i] > donchian_high_aligned[i]
        short_breakout = low[i] < donchian_low_aligned[i]
        trend_filter = adx_aligned[i] > 25
        vol_filter = vol_surge[i]
        
        long_entry = long_breakout and trend_filter and vol_filter
        short_entry = short_breakout and trend_filter and vol_filter
        
        # Exit conditions: price returns to middle of Donchian channel
        donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        exit_long = position == 1 and low[i] < donchian_mid
        exit_short = position == -1 and high[i] > donchian_mid
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals