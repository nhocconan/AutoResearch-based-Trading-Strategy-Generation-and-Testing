#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1w EMA50 for major trend direction (bullish if price > EMA50, bearish if price < EMA50).
- Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50 AND volume > 1.5 * 20-period volume MA.
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA50 AND volume > 1.5 * 20-period volume MA.
- Exit: Opposite Alligator alignment or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams Alligator identifies trend initiation and continuation, effective in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA)"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    result = np.empty_like(series, dtype=float)
    result[:] = np.nan
    # First value is simple SMA
    result[period-1] = np.mean(series[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 12h
    # Jaw: 13-period SMMA smoothed by 8 periods
    jaw = smma(close, 13)
    jaw = smma(jaw, 8)  # Additional smoothing
    
    # Teeth: 8-period SMMA smoothed by 5 periods
    teeth = smma(close, 8)
    teeth = smma(teeth, 5)  # Additional smoothing
    
    # Lips: 5-period SMMA smoothed by 3 periods
    lips = smma(close, 5)
    lips = smma(lips, 3)  # Additional smoothing
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 1w
    volume_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(volume_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_20_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for Alligator and 1w indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: bullish alignment AND price > 1w EMA50
                if bullish_alignment and curr_close > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: bearish alignment AND price < 1w EMA50
                elif bearish_alignment and curr_close < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: bearish alignment OR loss of volume confirmation
            if bearish_alignment or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR loss of volume confirmation
            if bullish_alignment or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0