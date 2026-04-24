#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d Elder Ray trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d Elder Ray (Bear/Bull Power) for trend direction (bullish when Bull Power > 0, bearish when Bear Power < 0).
- Entry: Price above Alligator's Jaw (long) or below Jaw (short) with Elder Ray alignment and volume > 2.0 * 12h volume MA(30).
- Exit: Price crosses Alligator's Teeth (midline) or opposite Jaw touch.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Alligator identifies trend absence/presence, Elder Ray confirms bull/bear power, volume filters weak breakouts.
- Works in bull markets by following Alligator alignment, works in bear markets by fading weak rallies against trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Get 1d data for Elder Ray and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h
    # Jaw (blue): 13-period SMMA, smoothed by 8 periods
    # Teeth (red): 8-period SMMA, smoothed by 5 periods  
    # Lips (green): 5-period SMMA, smoothed by 3 periods
    # Using EMA approximation for SMMA as it's similar and faster
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Jaw: 13 EMA + 8 EMA
    jaw = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: 8 EMA + 5 EMA
    teeth = pd.Series(close_12h).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: 5 EMA + 3 EMA
    lips = pd.Series(close_12h).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Align Alligator components from 12h to 12h (no shift needed as same TF)
    jaw_aligned = jaw  # Already on 12h
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema_13_1d
    bear_power = ema_13_1d - df_1d['low'].values
    
    # Align Elder Ray from 1d to 12h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 12h volume MA(30) for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
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
            vol_confirmed = curr_volume > 2.0 * vol_ma_12h_aligned[i]
            
            # Determine 1d Elder Ray trend: bullish if Bull Power > 0, bearish if Bear Power > 0
            trend_bullish = bull_power_aligned[i] > 0
            trend_bearish = bear_power_aligned[i] > 0
            
            # Long: price above Jaw AND Bull Power > 0 AND volume confirmed
            if curr_close > jaw_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price below Jaw AND Bear Power > 0 AND volume confirmed
            elif curr_close < jaw_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on close below Teeth (trend weakening) or touch Lips (mean reversion)
            if curr_close < teeth_aligned[i] or curr_close < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on close above Teeth (trend weakening) or touch Lips (mean reversion)
            if curr_close > teeth_aligned[i] or curr_close > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dElderRay_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0