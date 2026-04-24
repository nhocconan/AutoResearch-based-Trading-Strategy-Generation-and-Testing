#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot (H3/L3) breakout with 1w volume spike and ADX regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for volume average and trend direction, 1d for ADX regime filter.
- Camarilla Pivot: calculates key support/resistance levels from prior 1d OHLC.
- Entry: Long when price breaks above H3 AND volume > 2.0 * 1w average volume AND ADX(14) > 25 (trending regime).
         Short when price breaks below L3 AND volume > 2.0 * 1w average volume AND ADX(14) > 25.
- Exit: Opposite Camarilla breakout signal (price re-enters the pivot range).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets as ADX filter ensures we only trade strong trends,
  while Camarilla levels provide precise entry points after consolidation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def adx(high, low, close, period=14):
    """Calculate Average Directional Index with proper min_periods."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    
    # Directional Movement
    dm_plus = high_series.diff()
    dm_minus = low_series.diff() * -1
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM
    dm_plus_smooth = dm_plus.ewm(span=period, adjust=False, min_periods=period).mean()
    dm_minus_smooth = dm_minus.ewm(span=period, adjust=False, min_periods=period).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr.replace(0, 1e-10)
    di_minus = 100 * dm_minus_smooth / atr.replace(0, 1e-10)
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, 1e-10)
    adx_values = dx.ewm(span=period, adjust=False, min_periods=period).mean().values
    return adx_values

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    adx_1d = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Camarilla levels from 1d data (H3, L3, H4, L4)
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    # H4 = close + 1.1*(high-low)/2
    # L4 = close - 1.1*(high-low)/2
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    camarilla_high = df_1d['close'].values + 1.1 * (df_1d['high'].values - df_1d['low'].values) / 4
    camarilla_low = df_1d['close'].values - 1.1 * (df_1d['high'].values - df_1d['low'].values) / 4
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30)  # Need 20 for volume MA, 30 for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: price re-enters the Camarilla pivot range (H3 to L3)
        if position != 0:
            # Exit long: price moves back below H3
            if position == 1:
                if curr_close < camarilla_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price moves back above L3
            elif position == -1:
                if curr_close > camarilla_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and ADX regime filter
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= camarilla_high_aligned[i] and prev_close < camarilla_high_aligned[i-1]
            breakout_down = curr_low <= camarilla_low_aligned[i] and prev_close > camarilla_low_aligned[i-1]
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 2.0 * vol_ma_20_1w_aligned[i] if not np.isnan(vol_ma_20_1w_aligned[i]) else False
            
            # ADX regime filter: ADX(14) > 25 (trending regime)
            adx_regime = adx_1d_aligned[i] > 25
            
            if breakout_up and volume_confirm and adx_regime:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and adx_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wVolumeSpike_ADXRegime_v1"
timeframe = "12h"
leverage = 1.0