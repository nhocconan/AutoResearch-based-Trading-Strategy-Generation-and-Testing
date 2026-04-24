#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d volume spike and chop regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for Camarilla pivot calculation (based on prior day OHLC), volume average and choppiness index.
- Camarilla Pivots: identifies key support/resistance levels from prior 1d range.
- Entry: Long when price breaks above R1 AND volume > 1.8 * 20-period average volume AND CHOP(14) > 61.8 (range regime).
         Short when price breaks below S1 AND volume > 1.8 * 20-period average volume AND CHOP(14) > 61.8.
- Exit: Opposite Camarilla breakout (price crosses back below R1 for longs, above S1 for shorts).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla breakouts capture mean-reversion bounces from key levels in ranging markets.
- Volume confirmation ensures breakout legitimacy.
- Choppiness regime filter avoids trending markets where mean reversion fails.
- Works in both bull and bear markets as it captures volatility expansion from range contractions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def choppiness_index(high, low, close, period):
    """Calculate Choppiness Index with proper min_periods."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_series.rolling(window=period, min_periods=period).max()
    ll = low_series.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    # Handle division by zero when hh == ll
    chop = np.where((hh - ll) == 0, 50.0, chop.values)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivots (R1, S1) from prior 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for prior day calculation
        return np.zeros(n)
    
    # Prior day OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d Choppiness Index for regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    chop_14_1d = choppiness_index(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        14
    )
    chop_14_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: price crosses back below R1 for longs, above S1 for shorts
        if position != 0:
            # Exit long: price crosses below R1
            if position == 1:
                if curr_close < camarilla_r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above S1
            elif position == -1:
                if curr_close > camarilla_s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and chop regime filter
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= camarilla_r1_aligned[i] and prev_close < camarilla_r1_aligned[i-1]
            breakout_down = curr_low <= camarilla_s1_aligned[i] and prev_close > camarilla_s1_aligned[i-1]
            
            # Volume confirmation: current volume > 1.8 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.8 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Chop regime filter: CHOP(14) > 61.8 (range regime)
            chop_regime = chop_14_1d_aligned[i] > 61.8
            
            if breakout_up and volume_confirm and chop_regime:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and chop_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dVolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0