#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d ATR filter and volume spike confirmation.
- Primary timeframe: 4h for lower trade frequency and better signal quality.
- HTF: 1d ATR for volatility filter (only trade when ATR > 20-period ATR MA to avoid low volatility chop).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to capture institutional interest.
- Camarilla: H3 and L3 levels calculated from prior day's range.
- Entry: Long when price breaks above H3 AND 1d ATR > ATR MA AND volume spike.
         Short when price breaks below L3 AND 1d ATR > ATR MA AND volume spike.
- Exit: Price reverts to prior day's close (pivot) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe.
This strategy combines institutional volume confirmation with Camarilla pivot breakouts,
filtered by daily volatility to avoid ranging markets. Works in both bull and bear markets
by taking breakout trades with volume confirmation, avoiding low volatility false breakouts.
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
    
    # Get 1d data for Camarilla levels, ATR, and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR (14-period)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(df_1d_high[1:] - df_1d_low[1:])
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Prepend first TR as high-low for simplicity
    tr = np.concatenate([np.array([df_1d_high[0] - df_1d_low[0]]), tr])
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period 1d ATR MA for volatility filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior day's OHLC
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # Pivot (close) = (high + low + close) / 3
    h1d = df_1d['high'].values
    l1d = df_1d['low'].values
    c1d = df_1d['close'].values
    
    camarilla_h3 = c1d + 1.1 * (h1d - l1d) / 4
    camarilla_l3 = c1d - 1.1 * (h1d - l1d) / 4
    camarilla_pivot = (h1d + l1d + c1d) / 3  # Typical price as pivot
    
    # Align HTF indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volatility filter: 1d ATR > 20-period ATR MA
    volatility_filter = atr_1d_aligned > atr_ma_1d_aligned
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough bars for ATR14 + ATR MA20 + Vol MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(volume_spike[i]) or
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
            # Check for breakout signals with volume spike and volatility filter
            if volume_spike[i] and volatility_filter[i]:
                # Bullish breakout: price > H3
                if curr_close > camarilla_h3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < L3
                elif curr_close < camarilla_l3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to pivot OR loss of volume confirmation OR loss of volatility
            if curr_close <= camarilla_pivot_aligned[i] or not volume_spike[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to pivot OR loss of volume confirmation OR loss of volatility
            if curr_close >= camarilla_pivot_aligned[i] or not volume_spike[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0