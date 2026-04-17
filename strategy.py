#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Uses 1w pivot levels (PP, R1, S1) derived from previous week's OHLC.
# Direction filter: long only when price above weekly PP, short only when below weekly PP.
# Entry: breakout above/below Donchian(20) channel with volume spike and aligned with weekly pivot bias.
# Exit: reversal signal or Donchian midpoint retracement.
# Designed to capture institutional breakouts with low turnover (target: 12-37 trades/year).
# Weekly pivot bias ensures alignment with longer-term trend, reducing whipsaw in choppy markets.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (PP, R1, S1) from previous week
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    pp_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = (2 * pp_1w) - low_1w
    s1_1w = (2 * pp_1w) - high_1w
    
    # Calculate Donchian(20) channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2
    
    # Align weekly pivot levels to 6h timeframe
    pp_1w_6h = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1w_6h[i]) or 
            np.isnan(r1_1w_6h[i]) or 
            np.isnan(s1_1w_6h[i]) or 
            np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Price relative to weekly pivot
        price_above_pp = close[i] > pp_1w_6h[i]
        price_below_pp = close[i] < pp_1w_6h[i]
        
        # Price relative to Donchian channels
        breakout_up = close[i] > high_max_20[i]
        breakout_down = close[i] < low_min_20[i]
        
        if position == 0:
            # Long: Donchian breakout up with volume and above weekly PP
            if (breakout_up and price_above_pp and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down with volume and below weekly PP
            elif (breakout_down and price_below_pp and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian breakdown below midpoint OR reversal below weekly PP
            if (close[i] < donchian_mid[i]) or (close[i] < pp_1w_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian breakout above midpoint OR reversal above weekly PP
            if (close[i] > donchian_mid[i]) or (close[i] > pp_1w_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPP_Volume"
timeframe = "6h"
leverage = 1.0