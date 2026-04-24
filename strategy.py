#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for Camarilla levels, volume average, and choppiness calculation.
- Camarilla Pivots: calculated from 1d OHLC to identify key support/resistance levels.
- Entry: Long when price breaks above R1 AND volume > 1.8 * 20-period average volume AND CHOP(14) < 38.2 (trending regime).
         Short when price breaks below S1 AND volume > 1.8 * 20-period average volume AND CHOP(14) < 38.2.
- Exit: Opposite Camarilla breakout (close below S1 for long, close above R1 for short).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets as it captures institutional order flow at key levels with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC."""
    typical = (high + low + close) / 3
    range_ = high - low
    R4 = close + range_ * 1.1 / 2
    R3 = close + range_ * 1.1 / 4
    R2 = close + range_ * 1.1 / 6
    R1 = close + range_ * 1.1 / 12
    S1 = close - range_ * 1.1 / 12
    S2 = close - range_ * 1.1 / 6
    S3 = close - range_ * 1.1 / 4
    S4 = close - range_ * 1.1 / 2
    return R1, S1, R2, S2, R3, S3, R4, S4

def choppiness_index(high, low, close, period):
    """Calculate Choppiness Index to identify ranging vs trending markets."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over period
    sum_tr = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_series.rolling(window=period, min_periods=period).max()
    ll = low_series.rolling(window=period, min_periods=period).min()
    
    # Choppiness formula
    chop = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(period)
    chop_values = chop.values
    return chop_values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for meaningful pivots
        return np.zeros(n)
    
    # Calculate Camarilla for each 1d bar
    R1_1d = np.zeros(len(df_1d))
    S1_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        R1, S1, _, _, _, _, _, _ = calculate_camarilla(
            df_1d['high'].values[i],
            df_1d['low'].values[i],
            df_1d['close'].values[i]
        )
        R1_1d[i] = R1
        S1_1d[i] = S1
    
    # Align Camarilla levels to 4h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Calculate 1d volume average for confirmation
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d Choppiness Index for regime filter
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
    start_idx = 20  # Need 20 for volume MA alignment
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
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
        
        # Exit conditions: opposite Camarilla breakout (based on close)
        if position != 0:
            # Exit long: close below S1
            if position == 1:
                if curr_close <= S1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close above R1
            elif position == -1:
                if curr_close >= R1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and chop filter
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= R1_1d_aligned[i] and prev_close < R1_1d_aligned[i]
            breakout_down = curr_low <= S1_1d_aligned[i] and prev_close > S1_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.8 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.8 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Chop regime filter: CHOP < 38.2 (trending regime)
            chop_regime = chop_14_1d_aligned[i] < 38.2
            
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

name = "4h_Camarilla_R1S1_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0