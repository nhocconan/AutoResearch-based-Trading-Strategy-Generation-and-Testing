#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Price crosses above/below Alligator Jaw (13-period SMMA) with volume > 2.0 * 4h volume MA(50) and 1d EMA34 alignment.
- Exit: Price crosses back over Alligator Teeth (8-period SMMA) or opposite Jaw level.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Alligator identifies trend emergence/weakness, EMA34 filters higher-timeframe trend, volume confirms breakout validity.
- Works in bull markets by catching trends early, works in bear markets by fading counter-trend moves at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    if len(source) >= length:
        result[length-1] = np.nanmean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Alligator and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 4h Alligator components (SMMA)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    median_price_4h = (high_4h + low_4h) / 2.0  # Typical price for Alligator
    
    # Alligator Jaw (13-period SMMA of median price)
    jaw = smma(median_price_4h, 13)
    # Alligator Teeth (8-period SMMA of median price)
    teeth = smma(median_price_4h, 8)
    # Alligator Lips (5-period SMMA of median price)
    lips = smma(median_price_4h, 5)
    
    # Align Alligator levels from 4h to 4h timeframe (direct use with alignment for safety)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Calculate 1d EMA34 for trend
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 4h volume MA(50) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=50, min_periods=50).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma_4h_aligned[i]
            
            # Determine 1d EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
            trend_bullish = close[i] > ema_34_aligned[i]
            trend_bearish = close[i] < ema_34_aligned[i]
            
            # Alligator sleeping condition: Jaw, Teeth, Lips intertwined (Jaw between Teeth and Lips)
            # We avoid entries when Alligator is sleeping (no clear trend)
            jaw_val = jaw_aligned[i]
            teeth_val = teeth_aligned[i]
            lips_val = lips_aligned[i]
            is_sleeping = (jaw_val > teeth_val and jaw_val < lips_val) or (jaw_val < teeth_val and jaw_val > lips_val)
            
            # Long: price crosses above Jaw AND 1d trend bullish AND volume confirmed AND Alligator awakening (Jaw > Teeth)
            if curr_high > jaw_aligned[i] and trend_bullish and vol_confirmed and not is_sleeping and jaw_aligned[i] > teeth_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price crosses below Jaw AND 1d trend bearish AND volume confirmed AND Alligator awakening (Jaw < Teeth)
            elif curr_low < jaw_aligned[i] and trend_bearish and vol_confirmed and not is_sleeping and jaw_aligned[i] < teeth_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on cross below Teeth or touch of Lips (contrarian exit)
            if curr_close < teeth_aligned[i] or curr_low <= lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on cross above Teeth or touch of Lips (contrarian exit)
            if curr_close > teeth_aligned[i] or curr_high >= lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0