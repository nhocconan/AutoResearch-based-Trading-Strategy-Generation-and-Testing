#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot H3/L3 breakout with daily volume filter and ADX regime filter.
# Camarilla levels provide precise reversal points; H3 (resistance) and L3 (support) are key levels.
# Breakouts above H3 or below L3 with volume confirmation indicate strong momentum.
# ADX filter ensures we only trade in trending markets (ADX > 25) to avoid chop.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above H3) and bear markets (breakouts below L3).
name = "4h_Camarilla_H3L3_DailyVolume_ADX_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot and ADX (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (H3, L3) from previous day
    # Formula: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Calculate for each day, then shift to avoid look-ahead (use previous day's levels)
    camarilla_h3 = close_d + 1.1 * (high_d - low_d) / 6
    camarilla_l3 = close_d - 1.1 * (high_d - low_d) / 6
    
    # Shift by 1 to use previous day's levels
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_h3[0] = np.nan  # First day has no previous day
    camarilla_l3[0] = np.nan
    
    # Calculate ADX (14-period) for trend strength
    # +DM, -DM, TR calculation
    high_diff = np.diff(high_d, prepend=high_d[0])
    low_diff = -np.diff(low_d, prepend=low_d[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di = np.where(atr_14 != 0, 100 * plus_dm_14 / atr_14, 0)
    minus_di = np.where(atr_14 != 0, 100 * minus_dm_14 / atr_14, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align Camarilla levels and ADX to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 24-period average volume for confirmation (1 day of 4h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_24[i]
        
        # ADX filter: trending market (ADX > 25)
        adx_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above H3 AND volume confirmation AND ADX filter
            long_breakout = close[i] > camarilla_h3_aligned[i]
            if vol_confirm and adx_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND volume confirmation AND ADX filter
            elif vol_confirm and adx_filter and close[i] < camarilla_l3_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] < camarilla_l3_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H3 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] > camarilla_h3_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals