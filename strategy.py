#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
    # Long: price breaks above Camarilla H3 (1d) + volume > 1.3x 20-period avg + chop < 61.8
    # Short: price breaks below Camarilla L3 (1d) + volume > 1.3x 20-period avg + chop < 61.8
    # Exit: price returns to Camarilla pivot point (1d)
    # Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag
    # Camarilla pivots work well in both trending and ranging markets when combined with volume/regime filters
    # 4h timeframe provides good balance of signal quality and trade frequency
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Camarilla pivots and volume/chop filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate Camarilla levels on 1d data (based on previous day's OHLC)
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), etc.
    # L4 = close - 1.5*(high-low), L3 = close - 1.0*(high-low)
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.0 * range_1d
    camarilla_l3 = close_1d - 1.0 * range_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Chopiness Index on 1d data (14-period)
    def calculate_chop(high, low, close, window=14):
        # True Range
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        
        # Sum of True Range over window
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Highest high and lowest low over window
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        # Chop = log10(atr_sum / (highest_high - lowest_low)) / log10(window) * 100
        highest_low_diff = highest_high - lowest_low
        chop = np.where(
            (highest_low_diff > 0) & (~np.isnan(atr_sum)),
            np.log10(atr_sum / highest_low_diff) / np.log10(window) * 100,
            50  # default to middle when invalid
        )
        return chop
    
    chop = calculate_chop(high_1d, low_1d, close_1d, window=14)
    
    # Volume average on 1d data (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe (primary)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime filter: chop < 61.8 indicates trending market (good for breakouts)
        is_trending_regime = chop_aligned[i] < 61.8
        
        # Volume confirmation: current 1d volume > 1.3x 20-day average
        volume_confirmed = volume_1d[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Breakout conditions (using 4h price)
        breakout_up = close[i] > camarilla_h3_aligned[i]
        breakout_down = close[i] < camarilla_l3_aligned[i]
        
        # Entry conditions
        enter_long = is_trending_regime and breakout_up and volume_confirmed
        enter_short = is_trending_regime and breakout_down and volume_confirmed
        
        # Exit conditions: price returns to Camarilla pivot point
        exit_long = position == 1 and close[i] <= camarilla_pivot_aligned[i]
        exit_short = position == -1 and close[i] >= camarilla_pivot_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0