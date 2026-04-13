#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot fade with 1w trend filter and volume confirmation
    # Fade at H3/L3 (reversal) in ranging markets, breakout continuation at H4/L4 in trending markets
    # Uses 1w ADX to determine regime: ADX > 25 = trend (breakout), ADX < 20 = range (fade)
    # Volume confirmation reduces false signals
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 12-37 trades/year (~50-150 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for regime filter (ADX) and 1d for Camarilla pivots (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w ADX for regime filter (trend vs range)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1w index
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_period = 14
    atr_1w = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = wilders_smoothing(dx, atr_period)
    
    # Align 1w ADX to 6h (wait for completed 1w bar)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 1d Camarilla pivots (based on previous 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # PIVOT = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # RANGE = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels:
    # H3 = C + RANGE * 1.1/4, H4 = C + RANGE * 1.1/2
    # L3 = C - RANGE * 1.1/4, L4 = C - RANGE * 1.1/2
    h3_1d = close_1d + range_1d * 1.1 / 4
    h4_1d = close_1d + range_1d * 1.1 / 2
    l3_1d = close_1d - range_1d * 1.1 / 4
    l4_1d = close_1d - range_1d * 1.1 / 2
    
    # Align 1d Camarilla levels to 6h (wait for completed 1d bar)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume confirmation: >1.3x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Regime filter: ADX > 25 = trend (breakout), ADX < 20 = range (fade)
        is_trending = adx_1w_aligned[i] > 25
        is_ranging = adx_1w_aligned[i] < 20
        
        # Entry logic based on regime
        # In trending markets: breakout continuation at H4/L4
        # In ranging markets: fade at H3/L3 (mean reversion)
        long_entry = False
        short_entry = False
        
        if is_trending:
            # Trend: breakout continuation
            long_entry = (close[i] > h4_1d_aligned[i]) and vol_confirm
            short_entry = (close[i] < l4_1d_aligned[i]) and vol_confirm
        elif is_ranging:
            # Range: fade at H3/L3
            long_entry = (close[i] < l3_1d_aligned[i]) and vol_confirm  # fade from resistance
            short_entry = (close[i] > h3_1d_aligned[i]) and vol_confirm  # fade from support
        
        # Exit logic: return to pivot (mean reversion) or opposite signal
        long_exit = close[i] < pivot_1d_aligned[i]
        short_exit = close[i] > pivot_1d_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "6h_1w_1d_camarilla_regime_fade_v1"
timeframe = "6h"
leverage = 1.0