#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Williams Alligator: Jaw (13-period SMMA, 8-bar offset), Teeth (8-period SMMA, 5-bar offset), Lips (5-period SMMA, 3-bar offset).
- Entry: Long when Lips > Teeth > Jaw (Alligator bullish alignment) in 12h bull trend with volume > 1.5 * 4h volume MA(20); Short when Lips < Teeth < Jaw (Alligator bearish alignment) in 12h bear trend with volume > 1.5 * 4h volume MA(20).
- Exit: ATR-based trailing stop (2.0 * ATR(14)) or opposite Alligator alignment.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Williams Alligator identifies trending vs ranging markets, EMA50 filter ensures trend alignment, volume confirmation avoids false breakouts, works in both bull and bear markets via trend-following logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA)"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    smma = np.full_like(source, np.nan, dtype=float)
    smma[period-1] = np.mean(source[:period])
    for i in range(period, len(source)):
        smma[i] = (smma[i-1] * (period-1) + source[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator calculation and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Williams Alligator components on 4h
    jaw = smma(df_4h['close'].values, 13)  # Jaw: 13-period SMMA
    teeth = smma(df_4h['close'].values, 8)   # Teeth: 8-period SMMA
    lips = smma(df_4h['close'].values, 5)    # Lips: 5-period SMMA
    
    # Apply offsets as per Williams Alligator definition
    jaw = np.roll(jaw, 8)   # 8-bar offset
    teeth = np.roll(teeth, 5) # 5-bar offset
    lips = np.roll(lips, 3)   # 3-bar offset
    
    # Align Alligator components to 4h timeframe (already aligned via get_htf_data)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = vol_ma_4h  # already aligned
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14) + 8  # account for Alligator offsets
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirmed = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        
        # Determine 12h EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        # Williams Alligator alignment
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        
        alligator_bullish = lips_val > teeth_val > jaw_val
        alligator_bearish = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Check for entry signals
            # Long: Alligator bullish alignment in 12h bull trend with volume confirmation
            if alligator_bullish and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: Alligator bearish alignment in 12h bear trend with volume confirmation
            elif alligator_bearish and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite Alligator alignment
            if curr_low <= highest_since_entry - 2.0 * atr[i] or not alligator_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite Alligator alignment
            if curr_high >= lowest_since_entry + 2.0 * atr[i] or not alligator_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0