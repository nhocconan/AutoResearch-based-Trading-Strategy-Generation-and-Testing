#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla pivot H3/L3 breakout with 1w volume confirmation and ADX trend filter.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for volume average and ADX calculation.
- Camarilla Pivot: identifies key intraday support/resistance levels from prior day.
- Entry: Long when price breaks above H3 level AND volume > 1.5 * 1w average volume AND ADX(14) > 25 (trending regime).
         Short when price breaks below L3 level AND volume > 1.5 * 1w average volume AND ADX(14) > 25.
- Exit: Opposite Camarilla breakout signal or ADX drops below 20 (trend exhaustion).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels work well in both bull and bear markets as they adapt to recent volatility.
- Volume confirmation ensures breakout legitimacy.
- ADX filter avoids ranging markets where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC data."""
    typical = (high + low + close) / 3.0
    range_val = high - low
    h3 = typical + range_val * 1.1 / 4.0
    l3 = typical - range_val * 1.1 / 4.0
    return h3, l3

def adx(high, low, close, period=14):
    """Calculate Average Directional Index with proper min_periods."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Calculate True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    
    # Calculate Directional Movement
    up_move = high_series - high_series.shift(1)
    down_move = low_series.shift(1) - low_series
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_values = dx.ewm(span=period, adjust=False, min_periods=period).mean()
    
    return adx_values.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
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
    
    vol_ma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    # Calculate 1w ADX for trend filter
    if len(df_1w) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    adx_1w = adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate daily Camarilla levels from prior day
    camarilla_period = 1
    camarilla_high = pd.Series(high).rolling(window=camarilla_period, min_periods=camarilla_period).max().values
    camarilla_low = pd.Series(low).rolling(window=camarilla_period, min_periods=camarilla_period).min().values
    camarilla_close = pd.Series(close).rolling(window=camarilla_period, min_periods=camarilla_period).mean().values
    
    h3_levels = np.zeros(n)
    l3_levels = np.zeros(n)
    
    for i in range(n):
        if i < camarilla_period:
            h3_levels[i] = np.nan
            l3_levels[i] = np.nan
        else:
            idx = i - camarilla_period
            h3, l3 = calculate_camarilla(
                camarilla_high[idx],
                camarilla_low[idx],
                camarilla_close[idx]
            )
            h3_levels[i] = h3
            l3_levels[i] = l3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30, camarilla_period)  # Need 20 for volume MA, 30 for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_1w_aligned[i]) or
            np.isnan(h3_levels[i]) or np.isnan(l3_levels[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla breakout or trend exhaustion
        if position != 0:
            # Exit long: price breaks below L3 level OR ADX drops below 20 (trend exhaustion)
            if position == 1:
                if curr_low <= l3_levels[i] or adx_1w_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 level OR ADX drops below 20 (trend exhaustion)
            elif position == -1:
                if curr_high >= h3_levels[i] or adx_1w_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and ADX trend filter
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= h3_levels[i]
            breakout_down = curr_low <= l3_levels[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # ADX trend filter: ADX > 25 (strong trending regime)
            adx_trend = adx_1w_aligned[i] > 25
            
            if breakout_up and volume_confirm and adx_trend:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and adx_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wVolume_ADXTrend_v1"
timeframe = "1d"
leverage = 1.0