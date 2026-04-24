#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h to target 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3).
- Entry: Long when Lips > Teeth > Jaw (Alligator bullish) AND price > Lips AND 1d EMA50 bullish AND volume > 1.5 * volume MA(30).
         Short when Jaw > Teeth > Lips (Alligator bearish) AND price < Jaw AND 1d EMA50 bearish AND volume > 1.5 * volume MA(30).
- Exit: Close-based reversal - exit long when Lips < Teeth OR price < EMA50,
        exit short when Jaw > Teeth OR price > EMA50.
- Signal size: 0.25 discrete to balance return and drawdown.
Williams Alligator identifies trend initiation and continuation. Combined with 1d trend filter and volume confirmation,
it captures strong moves while avoiding choppy markets. Works in both bull and bear by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called Wilder's MA or RMA"""
    if length <= 0:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    alpha = 1.0 / length
    # First value is simple average
    if len(source) >= length:
        result[length-1] = np.mean(source[:length])
        # Subsequent values: SMMA = (prev_SMMA * (length-1) + current_price) / length
        for i in range(length, len(source)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply shifts (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set shifted values to NaN for invalid positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate volume MA(30) for confirmation
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Align HTF indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 13)  # Need enough bars for EMA50, Vol MA, and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Alligator bullish: Lips > Teeth > Jaw
            alligator_bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
            # Alligator bearish: Jaw > Teeth > Lips
            alligator_bearish = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
            
            # Long: Alligator bullish AND price > Lips AND 1d EMA50 bullish AND volume confirmed
            if alligator_bullish and curr_close > lips_aligned[i] and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND price < Jaw AND 1d EMA50 bearish AND volume confirmed
            elif alligator_bearish and curr_close < jaw_aligned[i] and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Lips < Teeth (Alligator turning bearish) OR price < EMA50 (trend change)
            if lips_aligned[i] < teeth_aligned[i] or curr_close < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Jaw > Teeth (Alligator turning bullish) OR price > EMA50 (trend change)
            if jaw_aligned[i] > teeth_aligned[i] or curr_close > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0