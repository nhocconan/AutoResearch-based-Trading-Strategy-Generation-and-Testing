#!/usr/bin/env python3
"""
6H_ADX_WILLIAMS_ALLIGATOR_COMBO
Hypothesis: Use 1d ADX for trend regime + Williams Alligator from 1d for entry timing.
- ADX > 25 indicates trending regime (trend following mode)
- In trending regime: Go long when price > Alligator's Jaw (13-period SMMA) and price > 8-period EMA
- Go short when price < Alligator's Jaw and price < 8-period EMA
- ADX <= 25 indicates ranging regime (mean reversion mode)
- In ranging regime: Go long when price touches Alligator's Lips (5-period SMMA) and RSI < 30
- Go short when price touches Alligator's Teeth (8-period SMMA) and RSI > 70
- Volume confirmation: require volume > 1.5x 20-period average to avoid fakeouts
- Target: 15-25 trades/year (60-100 total over 4 years) to stay within 6h limits.
Works in bull markets (trend following) and bear markets (mean reversion from extremes).
"""

name = "6H_ADX_WILLIAMS_ALLIGATOR_COMBO"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA) - same as used in Williams Alligator"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    result = np.full_like(series, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(series[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator from 1d
    # Jaw (13-period SMMA of median price)
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaw = smma(median_price.values, 13)
    # Teeth (8-period SMMA of median price)
    teeth = smma(median_price.values, 8)
    # Lips (5-period SMMA of median price)
    lips = smma(median_price.values, 5)
    
    # Align 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 8-period EMA on 6h close for entry timing
    close_series = pd.Series(close)
    ema8 = close_series.ewm(span=8, adjust=False, min_periods=8).values
    
    # RSI (14-period) on 6h for ranging regime entries
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(30, n):  # Start after warmup for ADX and SMMA
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or
            np.isnan(ema8[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Determine regime: ADX > 25 = trending, ADX <= 25 = ranging
            if adx_aligned[i] > 25:
                # TRENDING REGIME: Follow Alligator Jaw + EMA8
                # LONG: Price above Jaw AND above EMA8
                if close[i] > jaw_aligned[i] and close[i] > ema8[i]:
                    if volume_confirm[i]:
                        signals[i] = 0.25
                        position = 1
                        bars_since_entry = 0
                # SHORT: Price below Jaw AND below EMA8
                elif close[i] < jaw_aligned[i] and close[i] < ema8[i]:
                    if volume_confirm[i]:
                        signals[i] = -0.25
                        position = -1
                        bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                # RANGING REGIME: Mean reversion at Lips/Teeth with RSI extremes
                # LONG: Price at or below Lips AND RSI oversold
                if close[i] <= lips_aligned[i] and rsi[i] < 30:
                    if volume_confirm[i]:
                        signals[i] = 0.25
                        position = 1
                        bars_since_entry = 0
                # SHORT: Price at or above Teeth AND RSI overbought
                elif close[i] >= teeth_aligned[i] and rsi[i] > 70:
                    if volume_confirm[i]:
                        signals[i] = -0.25
                        position = -1
                        bars_since_entry = 0
                else:
                    signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: When price crosses below Jaw (trending) or reaches Teeth (ranging)
            if adx_aligned[i] > 25:
                # In trending regime, exit when price crosses below Jaw
                if close[i] < jaw_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging regime, exit when price reaches Teeth (mean reversion target)
                if close[i] >= teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: When price crosses above Jaw (trending) or reaches Lips (ranging)
            if adx_aligned[i] > 25:
                # In trending regime, exit when price crosses above Jaw
                if close[i] > jaw_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging regime, exit when price reaches Lips (mean reversion target)
                if close[i] <= lips_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals