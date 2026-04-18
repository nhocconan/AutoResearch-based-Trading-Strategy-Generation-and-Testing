#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with daily volume surge and ADX trend filter.
# Camarilla levels provide high-probability reversal/breakout points based on prior day's range.
# Volume surge confirms institutional participation in the breakout.
# ADX filter ensures we only trade in trending markets (ADX > 25) to avoid chop.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above H3/H4) and bear markets (breakdowns below L3/L4).
name = "4h_Camarilla_Pivot_Volume_ADX_Filter"
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
    
    # Get daily data for Camarilla calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low), etc.
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    range_hl = prev_high - prev_low
    H4 = prev_close + 1.5 * range_hl
    H3 = prev_close + 1.25 * range_hl
    L3 = prev_close - 1.25 * range_hl
    L4 = prev_close - 1.5 * range_hl
    
    # Align daily Camarilla levels to 4h timeframe (available after daily bar closes)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Calculate daily volume average for surge detection
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = df_1d['volume'].values > vol_ma_20  # Today's volume > 20-day average
    vol_surge_aligned = align_htf_to_ltf(prices, df_1d, vol_surge.astype(float))
    
    # Calculate ADX (14-period) on daily timeframe for trend filter
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_d[1:] - high_d[:-1]
    down_move = low_d[:-1] - low_d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA alpha=1/14)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = np.where((plus_di + minus_di) > 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_surge_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume surge: today's volume above 20-day average
        vol_confirm = vol_surge_aligned[i] > 0.5  # Boolean as float
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above H3 AND volume surge AND trending market
            long_breakout = close[i] > H3_aligned[i]
            if vol_confirm and trend_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND volume surge AND trending market
            elif vol_confirm and trend_filter and close[i] < L3_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below H3 (failed breakout) OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] < H3_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above L3 (failed breakdown) OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] > L3_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals