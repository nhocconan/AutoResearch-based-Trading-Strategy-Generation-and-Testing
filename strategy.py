#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d timeframe with volume confirmation.
# Long when price breaks above R4 level with volume > 1.2x 20-period average.
# Short when price breaks below S4 level with volume > 1.2x 20-period average.
# Exit when price crosses back below/above the central pivot (PP) or volume dries up.
# Uses daily Camarilla levels (calculated from prior day's H/L/C) for institutional support/resistance.
# Target: 80-180 total trades over 4 years (20-45/year) to stay within optimal range for 6h.

name = "6h_camarilla_pivot_1d_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation (yesterday's H/L/C)
    df_1d = get_htf_data(prices, '1d')
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (using prior day's data)
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # R2 = Close + 0.6 * (High - Low)
    # R1 = Close + 0.4 * (High - Low)
    # PP = (High + Low + Close) / 3
    # S1 = Close - 0.4 * (High - Low)
    # S2 = Close - 0.6 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_d_high = np.roll(d_high, 1)
    prev_d_low = np.roll(d_low, 1)
    prev_d_close = np.roll(d_close, 1)
    # First day has no previous data
    prev_d_high[0] = np.nan
    prev_d_low[0] = np.nan
    prev_d_close[0] = np.nan
    
    # Calculate levels
    camarilla_pp = (prev_d_high + prev_d_low + prev_d_close) / 3.0
    camarilla_range = prev_d_high - prev_d_low
    camarilla_r4 = prev_d_close + 1.5 * camarilla_range
    camarilla_s4 = prev_d_close - 1.5 * camarilla_range
    
    # Align to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume filter: current volume > 1.2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if Camarilla data not available
        if (np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.2
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below pivot OR volume dries up
            if (close[i] < camarilla_pp_aligned[i] or 
                not volume_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above pivot OR volume dries up
            if (close[i] > camarilla_pp_aligned[i] or 
                not volume_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: break above R4 level
                if high[i] > camarilla_r4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below S4 level
                elif low[i] < camarilla_s4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals