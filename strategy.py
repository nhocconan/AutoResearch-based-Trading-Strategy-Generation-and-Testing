#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R1/S1) breakout with daily volume confirmation and ADX trend filter.
# Camarilla levels from daily high/low provide institutional reversal points.
# Volume confirmation ensures breakout conviction.
# ADX filter ensures we only trade in trending markets (ADX > 25) to avoid chop.
# Works in bull markets (breakouts above R1) and bear markets (breakdowns below S1).
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
name = "12h_Camarilla_R1S1_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla and ADX (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    # Formula: Range = (High - Low), then levels based on Close
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_d, 1)
    prev_low = np.roll(low_d, 1)
    prev_close = np.roll(close_d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot and Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels: R1 = Close + (Range * 1.1/12), S1 = Close - (Range * 1.1/12)
    r1 = prev_close + (range_val * 1.1 / 12)
    s1 = prev_close - (range_val * 1.1 / 12)
    
    # Align daily Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate ADX (14-period) on daily data
    # +DM, -DM, TR calculation
    high_diff = np.diff(prev_high, prepend=np.nan)
    low_diff = np.diff(prev_low, prepend=np.nan)
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - np.roll(prev_close, 1))
    tr3 = np.abs(prev_low - np.roll(prev_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First element has no previous close
    
    # Smooth with Wilder's smoothing (EMA with alpha=1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align daily ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 24-period average volume for confirmation (2 days of 12h data)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
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
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirmation AND trend filter
            long_breakout = close[i] > r1_aligned[i]
            if vol_confirm and trend_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume confirmation AND trend filter
            elif vol_confirm and trend_filter and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] < s1_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] > r1_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals