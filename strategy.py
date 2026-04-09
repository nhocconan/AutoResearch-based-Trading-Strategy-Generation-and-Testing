#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels + volume confirmation + chop regime filter
# Camarilla pivots from 1d provide key support/resistance for swing trading
# Long when price breaks above H3 with volume confirmation in trending/regime conditions
# Short when price breaks below L3 with volume confirmation
# Uses Chop index to avoid ranging markets, Donchian exit for trend following
# Designed for 12h timeframe to target 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: breakout follows trends, chop filter avoids whipsaws in sideways markets

name = "12h_1d_camarilla_breakout_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    # H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low)
    # L3 = close - 1.25*(high-low), L4 = close - 1.5*(high-low)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.25 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.25 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align 1d indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute Chop index regime filter (14-period)
    # Chop > 61.8 = ranging (avoid), Chop < 38.2 = trending (favor breakouts)
    true_range = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    true_range[0] = high[0] - low[0]  # first bar
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_filter = chop < 50  # Avoid strong ranging (Chop > 50), favor trending/choppy
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price falls below L3 level (stop/reversal)
            if close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above H3 level (stop/reversal)
            if close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on Camarilla breakout with volume and chop confirmation
            if (close[i] > camarilla_h3_aligned[i] and volume_confirmed[i] and chop_filter[i]):
                position = 1
                signals[i] = 0.25
            elif (close[i] < camarilla_l3_aligned[i] and volume_confirmed[i] and chop_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals