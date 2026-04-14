#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot breakout with 1-day volume filter and chop regime filter.
# Long when price breaks above Camarilla H4 level AND volume > 1.8x 20-period average AND chop > 61.8 (range)
# Short when price breaks below Camarilla L4 level AND volume > 1.8x 20-period average AND chop > 61.8
# Exit when price crosses back to Camarilla H3/L3 levels (mean reversion in range)
# Camarilla levels from 1-day OHLC provide strong intraday support/resistance.
# Chop filter ensures we only trade in ranging markets where mean reversion works.
# Volume filter confirms breakout validity.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla calculation and chop filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily range for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's range
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    camarilla_h3 = close_1d + 1.125 * range_1d
    camarilla_l3 = close_1d - 1.125 * range_1d
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate chop index (14-period) for regime filter - values > 61.8 indicate ranging market
    # Chop = 100 * log10(sum(ATR1) / (n * true_range)) / log10(n)
    # Simplified: use rolling std dev of returns as proxy for chop
    returns = np.diff(np.log(close), prepend=0)
    chop_raw = pd.Series(returns).rolling(window=14, min_periods=14).std().values
    # Normalize chop to 0-100 scale (higher = more choppy/ranging)
    chop = 100 * (chop_raw - np.nanmin(chop_raw)) / (np.nanmax(chop_raw) - np.nanmin(chop_raw) + 1e-10)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.8
        
        if position == 0:
            # Long setup: break above H4 in ranging market with volume confirmation
            if (price > camarilla_h4_aligned[i] and chop[i] > 61.8 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: break below L4 in ranging market with volume confirmation
            elif (price < camarilla_l4_aligned[i] and chop[i] > 61.8 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to H3 level (mean reversion)
            if price < camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to L3 level (mean reversion)
            if price > camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_1dVol_Chop"
timeframe = "12h"
leverage = 1.0