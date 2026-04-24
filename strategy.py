#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend direction (bull/bear filter) and volume spike confirmation.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price.
- Trend Filter: 1d EMA34 > 1d EMA89 = bull trend, < = bear trend.
- Volume Confirmation: 4h volume > 1.5 * 20-period average volume.
- Entry: Long when Alligator aligned bullish (Lips > Teeth > Jaw) AND bull trend AND volume confirmation.
         Short when Alligator aligned bearish (Lips < Teeth < Jaw) AND bear trend AND volume confirmation.
- Exit: Opposite Alligator alignment (Lips crosses Teeth).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets by catching trends, avoids whipsaws in ranging markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2
    
    # Williams Alligator components (SMAs on median price)
    jaw_period, jaw_shift = 13, 8   # Jaw: 13-period SMA, shifted 8 bars
    teeth_period, teeth_shift = 8, 5  # Teeth: 8-period SMA, shifted 5 bars
    lips_period, lips_shift = 5, 3    # Lips: 5-period SMA, shifted 3 bars
    
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(jaw_shift).values
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_shift).values
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_shift).values
    
    # Calculate 1d EMA34 and EMA89 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Trend: bull when EMA34 > EMA89, bear when EMA34 < EMA89
    trend_bull = ema34_1d > ema89_1d
    trend_bear = ema34_1d < ema89_1d
    
    # Align trend to 4h timeframe
    trend_bull_aligned = align_htf_to_ltf(prices, df_1d, trend_bull.astype(float))
    trend_bear_aligned = align_htf_to_ltf(prices, df_1d, trend_bear.astype(float))
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lips_period + lips_shift, jaw_period + jaw_shift, 89)  # Need Alligator ready + 1d EMA89
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(trend_bull_aligned[i]) or np.isnan(trend_bear_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Alligator alignment
        lips_gt_teeth = lips[i] > teeth[i]
        teeth_gt_jaw = teeth[i] > jaw[i]
        lips_lt_teeth = lips[i] < teeth[i]
        teeth_lt_jaw = teeth[i] < jaw[i]
        
        bullish_aligned = lips_gt_teeth and teeth_gt_jaw  # Lips > Teeth > Jaw
        bearish_aligned = lips_lt_teeth and teeth_lt_jaw  # Lips < Teeth < Jaw
        
        # Exit conditions: opposite Alligator alignment
        if position != 0:
            # Exit long: Alligator turns bearish
            if position == 1 and bearish_aligned:
                signals[i] = 0.0
                position = 0
                continue
            # Exit short: Alligator turns bullish
            elif position == -1 and bullish_aligned:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Alligator alignment with trend and volume filters
        if position == 0:
            # Long: Bullish Alligator AND bull trend AND volume confirmation
            long_condition = bullish_aligned and trend_bull_aligned[i] > 0.5 and volume_confirm
            
            # Short: Bearish Alligator AND bear trend AND volume confirmation
            short_condition = bearish_aligned and trend_bear_aligned[i] > 0.5 and volume_confirm
            
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

name = "4h_WilliamsAlligator_1dEMATrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0