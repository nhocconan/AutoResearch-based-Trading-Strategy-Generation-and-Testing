#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v1
# Daily timeframe with weekly Camarilla pivot levels, volume confirmation, and ADX trend filter.
# Captures breakouts from weekly support/resistance levels with institutional volume.
# Works in bull markets (breakouts above H3/H4) and bear markets (breakdowns below L3/L4).
# Uses ADX to filter ranging markets and avoid false signals.
# Target: 15-25 trades/year per symbol for low friction and high edge.

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Camarilla formulas (H3/L3 and H4/L4 levels)
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align to daily timeframe (already delayed by 1 week due to shift)
    h3_level = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    h4_level = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    # ADX trend filter: avoid ranging markets (ADX < 25)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period]) / period
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_period = 14
    tr_smooth = wilder_smooth(tr, atr_period)
    dm_plus_smooth = wilder_smooth(dm_plus, atr_period)
    dm_minus_smooth = wilder_smooth(dm_minus, atr_period)
    
    # Avoid division by zero
    dm_plus_smooth = np.where(tr_smooth == 0, 1e-10, dm_plus_smooth)
    dm_minus_smooth = np.where(tr_smooth == 0, 1e-10, dm_minus_smooth)
    
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    # Smooth DX to get ADX
    adx = wilder_smooth(dx, atr_period)
    adx_filter = adx >= 25  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or np.isnan(h4_level[i]) or np.isnan(l4_level[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and ADX filters
        if not (vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above H4 with volume and trend
        if close[i] > h4_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Long entry on H3 break in strong trend
        elif close[i] > h3_level[i] and adx[i] > 30 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L4 with volume and trend
        elif close[i] < l4_level[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Short entry on L3 breakdown in strong trend
        elif close[i] < l3_level[i] and adx[i] > 30 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < l3_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > h3_level[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals