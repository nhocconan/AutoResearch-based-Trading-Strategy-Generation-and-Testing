#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WilliamsAlligator_ElderRay_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    """
    12h Williams Alligator trend filter + Elder Ray power for entry timing.
    Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMMA.
    Trend: Lips > Teeth > Jaw = uptrend; Lips < Teeth < Jaw = downtrend.
    Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
    Entry: Bull Power > 0 in uptrend, Bear Power > 0 in downtrend.
    Exit: Trend reversal or power crosses zero.
    Volume filter: Volume > 1.5x 20-period average.
    Target: 15-25 trades/year on 12h timeframe.
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray EMA(13) calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA(13) on 1d close for Elder Ray
    close_1d = pd.Series(df_1d['close'].values)
    ema13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Need 12h data for Alligator components
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    
    # Smoothed Moving Average (SMMA) - equivalent to Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_12h, 13)  # Jaw (13-period SMMA)
    teeth = smma(median_12h, 8)  # Teeth (8-period SMMA)
    lips = smma(median_12h, 5)   # Lips (5-period SMMA)
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema13_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        # Determine trend from Williams Alligator
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Lips < Teeth < Jaw
        is_uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        is_downtrend = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray power calculations
        bull_power = high[i] - ema13_1d_aligned[i]  # High - EMA13
        bear_power = ema13_1d_aligned[i] - low[i]   # EMA13 - Low
        
        if position == 0:
            # Long: Uptrend + Bull Power > 0 + Volume confirmation
            if is_uptrend and bull_power > 0 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + Bear Power > 0 + Volume confirmation
            elif is_downtrend and bear_power > 0 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend reversal or Bull Power <= 0
            if not is_uptrend or bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend reversal or Bear Power <= 0
            if not is_downtrend or bear_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals