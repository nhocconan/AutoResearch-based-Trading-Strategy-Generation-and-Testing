#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot structure and volume confirmation
# - Long: price breaks above weekly Camarilla H3 + Donchian(20) upper + volume > 1.5x 20-period avg
# - Short: price breaks below weekly Camarilla L3 + Donchian(20) lower + volume > 1.5x 20-period avg
# - Exit: price returns to weekly Camarilla H4/L4 levels
# - Uses weekly Camarilla pivots from 1w timeframe for structural support/resistance
# - Combines price channel breakout (Donchian) with pivot levels for confluence
# - Volume confirmation ensures breakout validity
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Weekly pivot structure works across market regimes (bull/bear/range)
# - Donchian(20) captures medium-term trends on 6h timeframe

name = "6h_1w_camarilla_donchian_volume_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1w data ONCE before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w Camarilla levels (based on previous week's OHLC)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    camarilla_h4 = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_h3 = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_l3 = close_1w - (high_1w - low_1w) * 1.1 / 4
    camarilla_l4 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align 1w Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Donchian channels (20-period) on 6h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(l4_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Weekly Camarilla levels
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        h4_level = h4_aligned[i]
        l4_level = l4_aligned[i]
        
        # Donchian channels
        donchian_upper = highest_20[i]
        donchian_lower = lowest_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above weekly Camarilla H3 AND Donchian upper, with volume confirmation
        if close_price > h3_level and close_price > donchian_upper and vol_confirm:
            enter_long = True
        
        # Short breakout: price below weekly Camarilla L3 AND Donchian lower, with volume confirmation
        if close_price < l3_level and close_price < donchian_lower and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to weekly Camarilla H4
            exit_long = close_price >= h4_level
        elif position == -1:
            # Exit short if price returns to weekly Camarilla L4
            exit_short = close_price <= l4_level
        
        # Track entry price for potential future use
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals