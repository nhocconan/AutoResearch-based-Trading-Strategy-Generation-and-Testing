#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h for lower trade frequency (<150 total trades over 4 years).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3).
- Volume: Current 12h volume > 1.5 * 20-period volume MA to filter low-participation moves.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND 1d EMA50 bullish AND volume spike.
         Short when Lips < Teeth < Jaw (bearish alignment) AND 1d EMA50 bearish AND volume spike.
- Exit: When Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw) OR loss of volume confirmation.
- Signal size: 0.25 discrete to minimize fee churn and control drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This strategy combines trend-following with the Williams Alligator (excellent in trending markets),
filtered by daily trend to avoid counter-trend trades, with volume confirmation ensuring
institutional participation. Works in both bull and bear markets by only taking trades
in the direction of the 1d trend, avoiding whipsaws in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    result = np.empty_like(source)
    result[:] = np.nan
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Apply shifts (Alligator specific)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set shifted values to NaN where roll creates invalid data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align HTF indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # Need enough bars for EMA50, volume MA, and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Check for Alligator alignment signals with volume spike
            if volume_spike[i]:
                # Bullish alignment: Lips > Teeth > Jaw
                if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]:
                    # Additionally confirm with 1d EMA50 trend (bullish)
                    if curr_close > ema_1d_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                # Bearish alignment: Lips < Teeth < Jaw
                elif lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]:
                    # Additionally confirm with 1d EMA50 trend (bearish)
                    if curr_close < ema_1d_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks OR loss of volume confirmation
            bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
            if not bullish_alignment or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks OR loss of volume confirmation
            bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
            if not bearish_alignment or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0