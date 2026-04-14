#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with weekly pivot direction and volume confirmation
# Uses daily Camarilla levels (H4/L4) for breakouts and weekly pivot (PP) for trend filter
# Works in bull/bear by only taking breakouts in direction of weekly trend
# Volume filter reduces false breakouts in low volatility
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily Camarilla levels (based on previous day)
    # H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # where C, H, L are from previous day
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Shift to use previous day's levels (available at open)
    camarilla_h4 = np.roll(camarilla_h4, 1)
    camarilla_l4 = np.roll(camarilla_l4, 1)
    camarilla_h4[0] = np.nan
    camarilla_l4[0] = np.nan
    
    # Calculate weekly pivot point (PP) for trend filter
    # PP = (H + L + C) / 3 (weekly values)
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align daily Camarilla levels to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Align weekly PP to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Volume confirmation: volume > 1.5x average volume (24-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(2, 24)  # for Camarilla shift and volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(pp_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above H4 AND above weekly PP with volume filter
            if (price > camarilla_h4_aligned[i] and price > pp_1w_aligned[i] and 
                vol > 1.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below L4 AND below weekly PP with volume filter
            elif (price < camarilla_l4_aligned[i] and price < pp_1w_aligned[i] and 
                  vol > 1.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below L4 OR below weekly PP
            if price < camarilla_l4_aligned[i] or price < pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above H4 OR above weekly PP
            if price > camarilla_h4_aligned[i] or price > pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Camarilla_Breakout_WeeklyPP_Volume"
timeframe = "6h"
leverage = 1.0