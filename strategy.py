#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume > 1.5 * 20-period volume MA.
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume > 1.5 * 20-period volume MA.
- Exit: Opposite Alligator alignment or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Williams Alligator identifies trends by aligning multiple smoothed moving averages. Works in both bull and bear markets by catching strong trends while avoiding choppy periods through volume confirmation and HTF EMA filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.empty_like(values, dtype=float)
    result[:] = np.nan
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
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
    
    # Calculate Williams Alligator on 4h
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price = (high + low) / 2
    jaw_raw = smma(median_price, 13)
    jaw = np.roll(jaw_raw, 8)  # shift 8 periods ahead
    jaw[:8] = np.nan  # first 8 values invalid due to shift
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = smma(median_price, 8)
    teeth = np.roll(teeth_raw, 5)  # shift 5 periods ahead
    teeth[:5] = np.nan  # first 5 values invalid due to shift
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = smma(median_price, 5)
    lips = np.roll(lips_raw, 3)  # shift 3 periods ahead
    lips[:3] = np.nan  # first 3 values invalid due to shift
    
    # Get 1d data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_34_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA on 1d for volume confirmation
    df_1d_volume = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Volume confirmation: current 4h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_confirm = volume > (1.5 * vol_ma_20_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13+8, 8+5, 5+3, 34, 20)  # Need enough bars for Alligator and 1d indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        ema_1d_val = ema_34_1d_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if vol_confirm:
                # Bullish: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34
                if lips_val > teeth_val > jaw_val and curr_close > ema_1d_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34
                elif lips_val < teeth_val < jaw_val and curr_close < ema_1d_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: bearish Alligator alignment OR loss of volume confirmation
            if lips_val < teeth_val < jaw_val or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish Alligator alignment OR loss of volume confirmation
            if lips_val > teeth_val > jaw_val or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0