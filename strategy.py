#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter.
    # Donchian breakout captures momentum in both bull and bear markets.
    # Volume spike confirms institutional participation (avoids false breakouts).
    # Chop regime filter (Choppiness Index > 61.8) enables mean reversion in ranging markets.
    # Target: 75-200 total trades over 4 years = 19-50/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume MA and chop regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max/Min high-low over 14 periods
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(TR(14)) / (maxHH - minLL)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_denom = max_hh - min_ll
    chop_denom = np.where(chop_denom == 0, np.nan, chop_denom)
    chop_raw = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    chop = chop_raw  # values: 0-100, >61.8 = ranging, <38.2 = trending
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Align HTF indicators to 4h timeframe
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period MA (spike confirmation)
        volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > donchian_high[i]  # Break above upper channel
        breakout_short = close[i] < donchian_low[i]  # Break below lower channel
        
        # Chop regime filter: > 61.8 = ranging (mean revert), < 38.2 = trending (trend follow)
        chop_value = chop_aligned[i]
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        # Entry conditions:
        # In trending regime: breakout in direction of breakout
        # In ranging regime: mean reversion (fade breakout)
        long_entry = False
        short_entry = False
        
        if is_trending:
            # Trend following: buy breakouts, sell breakdowns
            long_entry = breakout_long and volume_filter
            short_entry = breakout_short and volume_filter
        elif is_ranging:
            # Mean reversion: sell rallies, buy dips (fade extreme moves)
            long_entry = breakout_short and volume_filter  # buy dips
            short_entry = breakout_long and volume_filter  # sell rallies
        
        # Exit conditions: price returns to opposite Donchian level
        long_exit = close[i] < donchian_low[i]
        short_exit = close[i] > donchian_high[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0