#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with daily Camarilla pivot reversal + volume confirmation + chop filter
# Long when price crosses above Camarilla H3 level with volume surge in choppy market (CHOP>61.8)
# Short when price crosses below Camarilla L3 level with volume surge in choppy market (CHOP>61.8)
# Exit on opposite H4/L4 touch or when CHOP<38.2 (trending market)
# Target: 100-180 total trades over 4 years (25-45/year) using mean reversion in ranging markets
# Uses 1d for Camarilla levels and chop filter, 4h for execution

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4, etc.
    # Actually standard: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # Simpler: range = high-low, H3 = close + 1.1*range/4, L3 = close - 1.1*range/4
    # H4 = close + 1.1*range/2, L4 = close - 1.1*range/2
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    range_1d = prev_high - prev_low
    camarilla_h3 = prev_close + 1.1 * range_1d / 4
    camarilla_l3 = prev_close - 1.1 * range_1d / 4
    camarilla_h4 = prev_close + 1.1 * range_1d / 2
    camarilla_l4 = prev_close - 1.1 * range_1d / 2
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR14)/(n * true_range)) / log10(n)
    # Simplified: high-low as true range proxy
    tr1 = high_1d - low_1d
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    n_tr = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr14 / n_tr) / np.log10(14)
    chop = np.where(n_tr == 0, 50, chop)  # avoid division by zero
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge condition
        volume_surge = volume[i] > 1.5 * vol_ma_20[i]
        
        # Choppy market condition (range-bound)
        choppy = chop_aligned[i] > 61.8
        
        # Camarilla level touches
        touch_h3 = close[i] >= camarilla_h3_aligned[i]
        touch_l3 = close[i] <= camarilla_l3_aligned[i]
        touch_h4 = close[i] >= camarilla_h4_aligned[i]
        touch_l4 = close[i] <= camarilla_l4_aligned[i]
        
        # Entry logic: touch H3/L3 + volume surge + choppy market
        long_entry = touch_h3 and volume_surge and choppy
        short_entry = touch_l3 and volume_surge and choppy
        
        # Exit conditions: touch H4/L4 or chop < 38.2 (trending)
        exit_long = position == 1 and (touch_h4 or chop_aligned[i] < 38.2)
        exit_short = position == -1 and (touch_l4 or chop_aligned[i] < 38.2)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_chop_volume_reversal_v1"
timeframe = "4h"
leverage = 1.0