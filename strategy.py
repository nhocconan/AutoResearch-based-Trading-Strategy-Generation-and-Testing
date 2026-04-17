#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_Filter
Hypothesis: Camarilla pivot levels provide strong support/resistance in ranging markets and act as breakout levels in trending markets.
Long when price breaks above R1 with volume confirmation and 12h uptrend.
Short when price breaks below S1 with volume confirmation and 12h downtrend.
Exit on opposite breakout or loss of momentum. Position size: ±0.25.
Designed to work in bull (breakouts) and bear (reversals at pivots).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for each bar using previous day's OHLC
    # We need daily OHLC to calculate Camarilla levels
    # Since we're on 4h timeframe, we'll use 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shifted by 1)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day has no previous day, set to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla R1 and S1 for each day
    camarilla_R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume confirmation: 20-period average on 4h
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema34_12h = close_series_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 34)  # volume MA20, EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_R1_aligned[i]
        breakout_short = close[i] < camarilla_S1_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume filter + 12h uptrend
            if breakout_long and volume_filter and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + 12h downtrend
            elif breakout_short and volume_filter and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakout below S1 or loss of momentum (price < R1)
            if close[i] < camarilla_S1_aligned[i] or close[i] < camarilla_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above R1 or loss of momentum (price > S1)
            if close[i] > camarilla_R1_aligned[i] or close[i] > camarilla_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0