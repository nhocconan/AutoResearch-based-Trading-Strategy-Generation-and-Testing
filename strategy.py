#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA trend filter (EMA34 > EMA89 = bull trend, EMA34 < EMA89 = bear trend).
- Williams Alligator: Jaw (EMA13 of median price, 8-bar offset), Teeth (EMA8 of median price, 5-bar offset), Lips (EMA5 of median price, 3-bar offset).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND bull trend AND volume > 1.5 * 20-period average volume.
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND bear trend AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Alligator alignment (Lips < Teeth for long exit, Lips > Teeth for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets by catching trends, avoids chop by requiring clear Alligator alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 and EMA89 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA34 and EMA89
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Trend: 1 = bull (EMA34 > EMA89), -1 = bear (EMA34 < EMA89), 0 = unclear
    trend_1d = np.where(ema34_1d > ema89_1d, 1, np.where(ema34_1d < ema89_1d, -1, 0))
    
    # Align trend to 12h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw: EMA13 of median price, 8-bar offset
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # 8-bar offset to the right
    jaw[:8] = np.nan  # First 8 values are invalid
    
    # Teeth: EMA8 of median price, 5-bar offset
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # 5-bar offset to the right
    teeth[:5] = np.nan  # First 5 values are invalid
    
    # Lips: EMA5 of median price, 3-bar offset
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # 3-bar offset to the right
    lips[:3] = np.nan  # First 3 values are invalid
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20)  # Need 13 for Jaw, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Alligator alignment
        if position != 0:
            # Exit long: Lips <= Teeth (bullish alignment broken)
            if position == 1:
                if lips[i] <= teeth[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Lips >= Teeth (bearish alignment broken)
            elif position == -1:
                if lips[i] >= teeth[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with trend and volume filters
        if position == 0:
            # Long: Bullish alignment AND price > Lips AND bull trend AND volume confirmation
            long_condition = (bullish_alignment and 
                            curr_close > lips[i] and
                            trend_1d_aligned[i] == 1 and
                            volume_confirm)
            
            # Short: Bearish alignment AND price < Lips AND bear trend AND volume confirmation
            short_condition = (bearish_alignment and 
                             curr_close < lips[i] and
                             trend_1d_aligned[i] == -1 and
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