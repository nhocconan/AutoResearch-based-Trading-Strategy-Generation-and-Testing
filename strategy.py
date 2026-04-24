#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for EMA trend.
- Williams Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3).
- Breakout: Close > Lips (long) or Close < Jaw (short) with volume > 2.0x 20-period volume MA.
- Trend filter: Only trade breakouts in direction of 1d EMA50 (long if close > EMA50, short if close < EMA50).
- Works in bull via buying Alligator alignment (Lips>Teeth>Jaw) breakouts, in bear via selling breakdowns in inverse alignment.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
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
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Invalidate shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h (no additional delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(jaw))}), jaw)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(teeth))}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(lips))}), lips)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # EMA50 + volume MA + Alligator Jaw
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Alligator breakout with volume spike and trend filter
            if volume_spike[i]:
                # Bullish alignment: Lips > Teeth > Jaw
                bullish_align = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
                # Bearish alignment: Jaw > Teeth > Lips
                bearish_align = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
                
                # Long breakout: close > Lips AND bullish alignment AND close > 1d EMA50 (uptrend)
                if bullish_align and close[i] > lips_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < Jaw AND bearish alignment AND close < 1d EMA50 (downtrend)
                elif bearish_align and close[i] < jaw_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Alligator mouth (close < Teeth) or opposite signal
            if close[i] < teeth_aligned[i]:  # Exit when price falls below Teeth
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Alligator mouth (close > Teeth) or opposite signal
            if close[i] > teeth_aligned[i]:  # Exit when price rises above Teeth
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0