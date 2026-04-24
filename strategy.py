#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA21 trend filter and volume spike confirmation.
- Primary timeframe: 4h to target 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA21 for trend direction (bullish if close > EMA21, bearish if close < EMA21).
- Volume: Current 4h volume > 1.8 * 20-period volume MA to capture institutional interest.
- Camarilla: H3 and L3 levels calculated from prior day's range.
- Entry: Long when price breaks above H3 AND 1d EMA21 bullish AND volume spike.
         Short when price breaks below L3 AND 1d EMA21 bearish AND volume spike.
- Exit: Price reverts to prior day's close (typical price) OR loss of volume confirmation.
- Signal size: 0.25 discrete to minimize fee churn.
- Designed to work in both bull and bear markets by only taking trades in the direction of the 1d trend,
  with volume spikes confirming institutional participation. Camarilla levels provide natural S/R for exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA21
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d EMA21 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior day's OHLC
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # Pivot (typical price) = (high + low + close) / 3
    h1d = df_1d['high'].values
    l1d = df_1d['low'].values
    c1d = df_1d['close'].values
    
    camarilla_h3 = c1d + 1.1 * (h1d - l1d) / 4
    camarilla_l3 = c1d - 1.1 * (h1d - l1d) / 4
    camarilla_pivot = (h1d + l1d + c1d) / 3  # Typical price as pivot
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(21, 20)  # Need enough bars for EMA21 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for breakout signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price > H3 AND 1d EMA21 bullish (close > EMA)
                if curr_close > camarilla_h3_aligned[i] and curr_close > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < L3 AND 1d EMA21 bearish (close < EMA)
                elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to pivot OR loss of volume confirmation
            if curr_close <= camarilla_pivot_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to pivot OR loss of volume confirmation
            if curr_close >= camarilla_pivot_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0