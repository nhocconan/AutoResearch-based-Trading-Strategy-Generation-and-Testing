#/usr/bin/env python3
# 1D_Williams_Alligator_1wTrend_Filter
# Hypothesis: Uses Williams Alligator (SMMA with offsets) on daily chart filtered by weekly trend (close > weekly EMA50).
# Enters long when green line (lips) crosses above red line (jaw) in uptrend.
# Enters short when red line (jaw) crosses above green line (lips) in downtrend.
# Uses weekly EMA50 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 15-30 trades per year on 1d timeframe with position size 0.25.

name = "1D_Williams_Alligator_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    smoothed = np.full_like(values, np.nan, dtype=float)
    if len(values) < period:
        return smoothed
    # First value is simple average
    smoothed[period-1] = np.mean(values[:period])
    # Subsequent values: (prev_smoothed * (period-1) + current_value) / period
    for i in range(period, len(values)):
        smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
    return smoothed

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on daily chart
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply shifts (forward shift means we need to look at past values)
    # Jaw shifted 8 bars: use jaw[i-8] for current bar i
    # Teeth shifted 5 bars: use teeth[i-5] for current bar i
    # Lips shifted 3 bars: use lips[i-3] for current bar i
    
    jaw_shifted = np.roll(jaw, 8)  # shift right by 8
    teeth_shifted = np.roll(teeth, 5)  # shift right by 5
    lips_shifted = np.roll(lips, 3)  # shift right by 3
    
    # Set NaN for invalid shifted values
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are valid
    start_idx = max(13, 8, 5) + 8  # jaw period + jaw shift
    
    for i in range(start_idx, n):
        if np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Alligator crossover signals
        lips_above_jaw = lips_shifted[i] > jaw_shifted[i]
        lips_below_jaw = lips_shifted[i] < jaw_shifted[i]
        lips_crossed_above_jaw = lips_above_jaw and (lips_shifted[i-1] <= jaw_shifted[i-1])
        jaw_crossed_above_lips = jaw_shifted[i] > lips_shifted[i] and (jaw_shifted[i-1] <= lips_shifted[i-1])
        
        if position == 0:
            # Long entry: lips crosses above jaw with price above weekly EMA50 (uptrend)
            if lips_crossed_above_jaw and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: jaw crosses above lips with price below weekly EMA50 (downtrend)
            elif jaw_crossed_above_lips and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: jaw crosses above lips (trend weakness)
            if jaw_crossed_above_lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: lips crosses above jaw (trend reversal)
            if lips_crossed_above_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals