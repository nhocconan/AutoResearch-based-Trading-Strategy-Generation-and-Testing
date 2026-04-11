#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Bollinger Band squeeze + Donchian breakout + volume confirmation.
# Long when price breaks above Donchian upper band during low volatility (BBW < 20th percentile) with volume > 1.5x average.
# Short when price breaks below Donchian lower band during low volatility with volume > 1.5x average.
# Exits when price returns to Donchian middle or volatility expands (BBW > 80th percentile).
# Designed for low trade frequency (~20-40/year) to minimize fee drag while capturing volatility breakouts.
# Works in both bull/bear markets by only trading breakouts during low volatility regimes.

name = "4h_1d_bb_squeeze_donchian_breakout_v1"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized bandwidth
    
    # Calculate Bollinger Band width percentile (20-day lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align daily Bollinger Band width percentile to 4h timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_max_20 + low_min_20) / 2
    
    # Calculate 4h average volume (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure indicators are valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility regime: low volatility (BBW < 20th percentile)
        low_vol = bb_width_percentile_aligned[i] < 20
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_avg_20[i]
        
        # Entry conditions: Donchian breakout during low volatility with volume confirmation
        long_entry = (high[i] > high_max_20[i] and low_vol and vol_filter)
        short_entry = (low[i] < low_min_20[i] and low_vol and vol_filter)
        
        # Exit conditions: 
        # 1. Price returns to Donchian middle
        # 2. Volatility expands (BBW > 80th percentile)
        vol_expand = bb_width_percentile_aligned[i] > 80
        return_to_middle = (
            (position == 1 and low[i] < donchian_middle[i]) or
            (position == -1 and high[i] > donchian_middle[i])
        )
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (return_to_middle or vol_expand):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (return_to_middle or vol_expand):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals