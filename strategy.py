#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for trend strength (EMA50).
- Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3).
- Trend filter: Price > 1w EMA50 for long bias, Price < 1w EMA50 for short bias.
- Entry: Long when Lips cross above Teeth AND price > 1w EMA50 AND volume > 1.5 * 20-period volume MA.
         Short when Lips cross below Teeth AND price < 1w EMA50 AND volume > 1.5 * 20-period volume MA.
- Exit: Opposite Alligator cross (Lips cross Teeth in opposite direction).
- Volume confirmation: current volume > 1.5 * 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's EMA with alpha=1/period"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.empty_like(values, dtype=float)
    result[:] = np.nan
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
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
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Median price = (High + Low) / 2
    median_price = (df_1d['high'] + df_1d['low']) / 2
    
    # Alligator components: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(median_price.values, 13)
    jaw = smma(jaw, 8)  # Smoothed again with period 8
    
    teeth = smma(median_price.values, 8)
    teeth = smma(teeth, 5)  # Smoothed again with period 5
    
    lips = smma(median_price.values, 5)
    lips = smma(lips, 3)  # Smoothed again with period 3
    
    # Align Alligator components to 1d timeframe (already aligned via get_htf_data)
    # But we need to align to the lower timeframe (1d prices index)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 1d)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 20)  # Need enough bars for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        price = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Check for Alligator crossovers
        lips_above_teeth = lips_val > teeth_val
        lips_below_teeth = lips_val < teeth_val
        
        # Previous values for crossover detection
        if i > 0:
            lips_prev = lips_aligned[i-1]
            teeth_prev = teeth_aligned[i-1]
            lips_above_teeth_prev = lips_prev > teeth_prev
            lips_below_teeth_prev = lips_prev < teeth_prev
            
            lips_cross_above = lips_above_teeth and not lips_above_teeth_prev
            lips_cross_below = lips_below_teeth and not lips_below_teeth_prev
        else:
            lips_cross_above = False
            lips_cross_below = False
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if vol_spike:
                # Long: Lips cross above Teeth AND price > 1w EMA50 (uptrend)
                if lips_cross_above and price > ema_50_val:
                    signals[i] = 0.25
                    position = 1
                # Short: Lips cross below Teeth AND price < 1w EMA50 (downtrend)
                elif lips_cross_below and price < ema_50_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Lips cross below Teeth (trend weakening)
            if lips_cross_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Lips cross above Teeth (trend weakening)
            if lips_cross_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0