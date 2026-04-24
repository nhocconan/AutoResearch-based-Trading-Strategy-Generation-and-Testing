#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h for lower frequency to minimize fee drag.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period smoothed median), Lips (5-period smoothed median).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND 1d EMA34 bullish AND volume spike.
         Short when Lips < Teeth < Jaw (bearish alignment) AND 1d EMA34 bearish AND volume spike.
- Volume: Current 12h volume > 2.0 * 24-period 12h volume MA to capture institutional interest.
- Exit: Opposite Alligator alignment or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This combines trend-following via Alligator alignment with higher-timeframe trend filtering and volume confirmation
to capture sustained moves in both bull and bear markets while avoiding choppy periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator components (Smoothed Moving Average = SMMA)
    def smma(data, period):
        # Smoothed Moving Average: first value is SMA, then recursive smoothing
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(data, np.nan, dtype=float)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(data)):
            if not np.isnan(smma_vals[i-1]) and not np.isnan(data[i]):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + data[i]) / period
            else:
                smma_vals[i] = np.nan
        return smma_vals
    
    jaw = smma(median_price, 13)  # Jaw (13-period SMMA of median price)
    teeth = smma(median_price, 8)  # Teeth (8-period SMMA of median price)
    lips = smma(median_price, 5)   # Lips (5-period SMMA of median price)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 24-period 12h volume MA
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align HTF indicators to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current 12h volume > 2.0 * 24-period 12h volume MA
    volume_spike = volume > (2.0 * vol_ma)
    
    # Alligator alignment signals
    lips_gt_teeth = lips_aligned > teeth_aligned
    teeth_gt_jaw = teeth_aligned > jaw_aligned
    bullish_align = lips_gt_teeth & teeth_gt_jaw  # Lips > Teeth > Jaw
    
    lips_lt_teeth = lips_aligned < teeth_aligned
    teeth_lt_jaw = teeth_aligned < jaw_aligned
    bearish_align = lips_lt_teeth & teeth_lt_jaw  # Lips < Teeth < Jaw
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 34, 13)  # Need enough bars for volume MA, EMA34, and Alligator jaw
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Alligator bullish alignment AND 1d EMA34 bullish (close > EMA)
                if bullish_align[i] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Alligator bearish alignment AND 1d EMA34 bearish (close < EMA)
                elif bearish_align[i] and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR loss of volume confirmation
            if bearish_align[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR loss of volume confirmation
            if bullish_align[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0