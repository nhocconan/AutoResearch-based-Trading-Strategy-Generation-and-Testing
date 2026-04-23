#!/usr/bin/env python3
"""
Hypothesis: 6-hour Bollinger Band width regime filter combined with 12-hour Donchian breakout and volume confirmation.
Long when price breaks above Donchian(20) high, BB width < 30th percentile (low volatility squeeze), and volume > 1.5x average.
Short when price breaks below Donchian(20) low, BB width < 30th percentile, and volume > 1.5x average.
Exit when price returns to Donchian middle or BB width > 70th percentile (high volatility).
Designed to catch breakouts from low volatility contractions in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for Donchian and BB width - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour Bollinger Bands (20, 2)
    close_12h = df_12h['close'].values
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Calculate 12-hour Donchian channels (20)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Align HTF indicators to lower timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_12h, bb_width)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Percentile thresholds for BB width (pre-compute to avoid look-ahead)
    bb_width_series = pd.Series(bb_width_aligned)
    bb_width_lower_thresh = bb_width_series.rolling(window=100, min_periods=100).quantile(0.30).values
    bb_width_upper_thresh = bb_width_series.rolling(window=100, min_periods=100).quantile(0.70).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(bb_width_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(donch_mid_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(bb_width_lower_thresh[i]) or 
            np.isnan(bb_width_upper_thresh[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bb_width_val = bb_width_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        donch_mid_val = donch_mid_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        bb_width_lower_thresh_val = bb_width_lower_thresh[i]
        bb_width_upper_thresh_val = bb_width_upper_thresh[i]
        
        if position == 0:
            # Long: Donchian breakout up, low volatility squeeze, volume confirmation
            if (close_val > donch_high_val and 
                bb_width_val < bb_width_lower_thresh_val and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, low volatility squeeze, volume confirmation
            elif (close_val < donch_low_val and 
                  bb_width_val < bb_width_lower_thresh_val and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to Donchian mid OR high volatility (BB width > 70th percentile)
                if (close_val < donch_mid_val or bb_width_val > bb_width_upper_thresh_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to Donchian mid OR high volatility
                if (close_val > donch_mid_val or bb_width_val > bb_width_upper_thresh_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_BBWidth_DonchianBreakout_Volume"
timeframe = "6h"
leverage = 1.0