#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d volume spike and ADX trend filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Camarilla pivot levels, volume average, and ADX calculation.
- Camarilla Pivots: identifies key support/resistance levels from prior 1d range.
- Entry: Long when price breaks above R3 with volume confirmation and ADX > 25 (trending).
         Short when price breaks below S3 with volume confirmation and ADX > 25.
- Exit: Opposite Camarilla breakout (R4/S4) or ADX < 20 (trend weakens).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla breakouts capture institutional order flow at key levels.
- Volume confirmation avoids false breakouts.
- ADX filter ensures we trade only in trending markets, avoiding chop.
- Works in bull markets (breakouts continuation) and bear markets (breakdown continuation).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def adx(high, low, close, period=14):
    """Average Directional Index with proper min_periods."""
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
    up_move = high_series - high_series.shift(1)
    down_move = low_series.shift(1) - low_series
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_values = dx.ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx_values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Calculate 1d volume average for confirmation (20-period MA)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d ADX for trend filter
    adx_1d = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla formula: based on previous day's range
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    pivot = (h_1d + l_1d + c_1d) / 3
    range_1d = h_1d - l_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 4)
    r4 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need 30 for ADX, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below R3 or ADX < 20 (trend weakening)
            if position == 1:
                if curr_low <= r3_aligned[i] or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above S3 or ADX < 20
            elif position == -1:
                if curr_high >= s3_aligned[i] or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and ADX trend filter
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= r3_aligned[i] and prev_close < r3_aligned[i-1]
            breakout_down = curr_low <= s3_aligned[i] and prev_close > s3_aligned[i-1]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # ADX trend filter: ADX > 25 (strong trend)
            adx_trend = adx_1d_aligned[i] > 25
            
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

name = "6h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADXTrend_v1"
timeframe = "6h"
leverage = 1.0