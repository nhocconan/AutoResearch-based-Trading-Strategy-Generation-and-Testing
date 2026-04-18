#!/usr/bin/env python3
"""
1h_4h_1d_MultiTF_Trend_Follow_With_Pullback
Hypothesis: Combine 4h EMA trend filter and 1d ADX trend strength with 1h pullback entries.
In bull markets (4h EMA20 rising + 1d ADX>25), buy pullbacks to 1h EMA20.
In bear markets (4h EMA20 falling + 1d ADX>25), sell rallies to 1h EMA20.
Uses multi-timeframe alignment to avoid whipsaws and capture trend moves with controlled frequency.
Designed for 15-30 trades/year to minimize fee drag while maintaining edge in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            result[period-1] = np.nansum(x[:period])
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    tr_smooth = smooth(tr, period)
    dm_plus_smooth = smooth(dm_plus, period)
    dm_minus_smooth = smooth(dm_minus, period)
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / tr_smooth
    minus_di = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth(dx, period)
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_prev = np.roll(ema_20_4h, 1)
    ema_20_4h_prev[0] = np.nan
    ema_20_4h_slope = ema_20_4h - ema_20_4h_prev  # Rising if >0, falling if <0
    ema_20_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h_slope)
    
    # Get 1d data for ADX trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d, additional_delay_bars=0)
    
    # Get 1h EMA20 for pullback entries
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    # Start after sufficient warmup for all indicators
    start_idx = max(50, 30)  # ADX and EMA warmups
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_4h_slope_aligned[i]) or 
            np.isnan(adx_14_1d_aligned[i]) or
            np.isnan(ema_20_1h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema4h_slope = ema_20_4h_slope_aligned[i]
        adx1d = adx_14_1d_aligned[i]
        ema1h = ema_20_1h[i]
        
        # Trend regime: need strong trend (ADX > 25) on daily
        is_strong_trend = adx1d > 25
        
        if position == 0:
            # Enter long: bullish trend (4h EMA20 rising) + pullback to 1h EMA20
            if is_strong_trend and ema4h_slope > 0 and price <= ema1h * 1.005:  # Within 0.5% of EMA
                signals[i] = 0.20
                position = 1
            # Enter short: bearish trend (4h EMA20 falling) + rally to 1h EMA20
            elif is_strong_trend and ema4h_slope < 0 and price >= ema1h * 0.995:  # Within 0.5% of EMA
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: trend weakness or opposite signal
            if not (is_strong_trend and ema4h_slope > 0):
                signals[i] = 0.0
                position = 0
            # Optional: exit on strong adverse move
            elif price < ema1h * 0.98:  # 2% below EMA
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: trend weakness or opposite signal
            if not (is_strong_trend and ema4h_slope < 0):
                signals[i] = 0.0
                position = 0
            # Optional: exit on strong adverse move
            elif price > ema1h * 1.02:  # 2% above EMA
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_4h_1d_MultiTF_Trend_Follow_With_Pullback"
timeframe = "1h"
leverage = 1.0