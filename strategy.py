# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_1d_williams_alligator_volume_v1
Strategy: 12h Williams Alligator with volume confirmation and 1d trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Williams Alligator (SMMA-based) identifies trend direction on 12h. Long when price above Alligator teeth (13-period SMMA) with volume confirmation and 1d uptrend. Short when price below teeth with volume confirmation and 1d downtrend. Uses Williams Alligator's smoothed moving averages to reduce whipsaw in ranging markets, combined with volume confirmation to avoid false breakouts. Designed for low-frequency, high-conviction trades targeting 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_williams_alligator_volume_v1"
timeframe = "12h"
leverage = 1.0

def _smma(arr, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's smoothing"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Price) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Williams Alligator on 12h timeframe (using SMMA)
    # Jaw: 13-period SMMA of median price, shifted 8 bars forward
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward  
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    median_price = (high + low) / 2
    
    jaw_raw = _smma(median_price, 13)
    teeth_raw = _smma(median_price, 8)
    lips_raw = _smma(median_price, 5)
    
    # Apply forward shifts (Alligator specific)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Invalidate the shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume confirmation threshold
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after Alligator warmup
        # Skip if any required data is invalid
        if (np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Alligator conditions: price above/below teeth with lips aligned
        # Bullish: price > teeth AND lips > teeth (alignment)
        # Bearish: price < teeth AND lips < teeth (alignment)
        bullish_alignment = price_close > teeth[i] and lips[i] > teeth[i]
        bearish_alignment = price_close < teeth[i] and lips[i] < teeth[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: bullish alignment with volume in uptrend
        long_signal = bullish_alignment and vol_confirmed and uptrend_1d
        
        # Short: bearish alignment with volume in downtrend
        short_signal = bearish_alignment and vol_confirmed and downtrend_1d
        
        # Exit when price crosses lips in opposite direction or loses alignment
        exit_long = position == 1 and (price_close < lips[i] or lips[i] < teeth[i])
        exit_short = position == -1 and (price_close > lips[i] or lips[i] > teeth[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals