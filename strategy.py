#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w EMA trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1w for trend direction.
- Williams Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3).
- Trend: 1w EMA34 > 1w EMA89 = bullish trend, 1w EMA34 < 1w EMA89 = bearish trend.
- Entry: Long when Alligator lines are bullish (Lips > Teeth > Jaw) AND price > Lips AND 1w bullish trend AND volume > 1.5 * 20-period volume MA.
         Short when Alligator lines are bearish (Lips < Teeth < Jaw) AND price < Lips AND 1w bearish trend AND volume > 1.5 * 20-period volume MA.
- Exit: Opposite Alligator signal or price crosses Jaw.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false signals).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.empty_like(data)
    result[:] = np.nan
    # First value is simple SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_value) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 and EMA89 for trend
    close_1w = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1w = close_1w.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1w EMA to 12h
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema89_1w)
    
    # Williams Alligator on 12h
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = smma((high + low) / 2, 13)
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = smma((high + low) / 2, 8)
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = smma((high + low) / 2, 5)
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1w bars for EMA and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(ema89_1w_aligned[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1w trend
        bullish_trend = ema34_1w_aligned[i] > ema89_1w_aligned[i]
        bearish_trend = ema34_1w_aligned[i] < ema89_1w_aligned[i]
        
        # Alligator alignment
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        
        bullish_alligator = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alligator = lips_val < teeth_val and teeth_val < jaw_val
        
        curr_close = close[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish entry: bullish Alligator + price above Lips + bullish 1w trend
                if bullish_alligator and curr_close > lips_val and bullish_trend:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: bearish Alligator + price below Lips + bearish 1w trend
                elif bearish_alligator and curr_close < lips_val and bearish_trend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below Jaw OR Alligator turns bearish
            if curr_close < jaw_val or not bullish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Jaw OR Alligator turns bullish
            if curr_close > jaw_val or not bearish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMATrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0