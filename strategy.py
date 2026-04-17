#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Camarilla R1/S1 breakout + volume confirmation + chop regime filter.
Long when price breaks above R1 with volume > 1.5x 20-period average and CHOP > 61.8 (ranging market).
Short when price breaks below S1 with volume > 1.5x 20-period average and CHOP > 61.8.
Exit when price returns to the 1d CAMARILLA PIVOT (PP) level.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Camarilla pivot levels provide precise intraday support/resistance; volume confirms breakout validity;
chop filter ensures we only trade in ranging markets where mean reversion at pivot levels works.
Designed to work in ranging markets (2025+ bear/range conditions) by fading false breakouts at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1, PP)
    # Camarilla: PP = (H + L + C)/3; R1 = C + (H-L)*1.1/12; S1 = C - (H-L)*1.1/12
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    def chop(high_vals, low_vals, close_vals, window):
        atr = pd.Series(np.maximum(np.maximum(high_vals - low_vals, 
                                             np.abs(high_vals - np.roll(close_vals, 1))), 
                                  np.abs(np.roll(close_vals, 1) - low_vals))).rolling(window=window, min_periods=1).mean()
        max_high = pd.Series(high_vals).rolling(window=window, min_periods=1).max()
        min_low = pd.Series(low_vals).rolling(window=window, min_periods=1).min()
        chop_val = 100 * np.log10(atr.sum() / np.log(window) / (max_high - min_low)) / np.log10(window)
        return chop_val.fillna(50).values  # neutral when undefined
    
    chop_1d = chop(high_1d, low_1d, close_1d, 14)
    
    # Align all to primary timeframe (12h)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for volume MA and chop calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        # Chop filter: CHOP > 61.8 indicates ranging market (mean reversion regime)
        ranging_market = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation in ranging market
            if (close[i] > r1_1d_aligned[i] and 
                volume_confirmed and 
                ranging_market):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation in ranging market
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_confirmed and 
                  ranging_market):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below pivot point (PP)
            if close[i] <= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above pivot point (PP)
            if close[i] >= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarilla_R1S1_Volume_Chop_Filter"
timeframe = "12h"
leverage = 1.0