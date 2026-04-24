#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend direction and volume spike filter.
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price.
- Trend: 1d EMA34 > 1d EMA89 = bull trend, < = bear trend.
- Volume: 12h volume > 1.5 * 20-period average volume for confirmation.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND bull trend AND volume confirmation.
         Short when Lips < Teeth < Jaw (bearish alignment) AND bear trend AND volume confirmation.
- Exit: Opposite Alligator alignment (Lips crosses Teeth).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by trading with the 1d trend, avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Calculate Williams Alligator components (SMAs of median price)
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d EMA34 and EMA89 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Trend: bull if EMA34 > EMA89, bear if EMA34 < EMA89
    trend_bull = ema34_1d > ema89_1d
    trend_bear = ema34_1d < ema89_1d
    
    # Align trend to 12h timeframe
    trend_bull_aligned = align_htf_to_ltf(prices, df_1d, trend_bull)
    trend_bear_aligned = align_htf_to_ltf(prices, df_1d, trend_bear)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13+8, 8+5, 5+3, 89, 20)  # Max of Alligator shifts, EMA89, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(trend_bull_aligned[i]) or np.isnan(trend_bear_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Alligator alignment conditions
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # Exit conditions: opposite Alligator alignment
        if position != 0:
            # Exit long: bearish alignment (Lips < Teeth < Jaw)
            if position == 1:
                if bearish_alignment:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish alignment (Lips > Teeth > Jaw)
            elif position == -1:
                if bullish_alignment:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with trend and volume filters
        if position == 0:
            # Long: bullish alignment AND bull trend AND volume confirmation
            long_condition = (bullish_alignment and 
                            trend_bull_aligned[i] and
                            volume_confirm)
            
            # Short: bearish alignment AND bear trend AND volume confirmation
            short_condition = (bearish_alignment and 
                             trend_bear_aligned[i] and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMATrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0