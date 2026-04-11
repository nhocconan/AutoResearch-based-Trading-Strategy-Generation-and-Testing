#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h/1d Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# - Choppiness Index (CHOP) > 61.8 = ranging market (mean reversion), CHOP < 38.2 = trending
# - In trending regime: Donchian(20) breakout with volume confirmation
# - In ranging regime: mean reversion at Bollinger Bands(20,2) with volume confirmation
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits for 4h
# - Works in both bull (trend continuation) and bear (trend reversal) markets
# - 1d Choppiness Index provides regime filter, reducing false signals in choppy markets

name = "4h_1d_chop_donchian_bbands_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Choppiness Index regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return signals
    
    # Pre-compute 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    chop_raw = 100 * np.log10(atr_sum / hl_range) / np.log10(14)
    chop_raw = np.where(hl_range == 0, 100, chop_raw)  # Set to 100 when range is zero
    chop_raw = np.where(np.isnan(chop_raw), 50, chop_raw)  # Default to middle when NaN
    
    chop_1d = chop_raw
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute 4h indicators
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Volume SMA (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Regime filter: Choppiness Index
        chop_value = chop_1d_aligned[i]
        is_trending = chop_value < 38.2  # Trending market
        is_ranging = chop_value > 61.8   # Ranging market
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Initialize signals
        enter_long = False
        enter_short = False
        exit_long = False
        exit_short = False
        
        if is_trending:
            # Trending regime: Donchian breakout
            # Long: price breaks above Donchian high + volume confirmation
            if price_high > donch_high[i] and vol_confirm:
                enter_long = True
            # Short: price breaks below Donchian low + volume confirmation
            if price_low < donch_low[i] and vol_confirm:
                enter_short = True
            # Exit: opposite Donchian breakout
            if position == 1 and price_low < donch_low[i]:
                exit_long = True
            if position == -1 and price_high > donch_high[i]:
                exit_short = True
                
        elif is_ranging:
            # Ranging regime: Bollinger Bands mean reversion
            # Long: price touches/below lower BB + volume confirmation
            if price_low <= bb_lower[i] and vol_confirm:
                enter_long = True
            # Short: price touches/above upper BB + volume confirmation
            if price_high >= bb_upper[i] and vol_confirm:
                enter_short = True
            # Exit: price returns to middle (SMA)
            if position == 1 and price_close >= sma_20[i]:
                exit_long = True
            if position == -1 and price_close <= sma_20[i]:
                exit_short = True
        
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