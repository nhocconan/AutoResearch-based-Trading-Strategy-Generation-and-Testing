#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and 1w ADX regime filter
    # Long: price breaks above H3 AND 1w ADX > 25 (strong trend) AND volume > 2.0x 24-period avg
    # Short: price breaks below L3 AND 1w ADX > 25 AND volume > 2.0x 24-period avg
    # Exit: price touches H4/L4 levels or retests H3/L3
    # Using 12h timeframe for optimal trade frequency (target 12-37/year), Camarilla for structure,
    # 1w ADX to filter weak trends, and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for trend strength filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align weekly ADX to 12h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, L3, L4
    # H4 = close + 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4
    # L4 = close - 1.1*(high-low)*1.1/2
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    for i in range(1, len(high_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        # Calculate levels for current day
        h4 = prev_close + 1.1 * range_val * 1.1 / 2
        h3 = prev_close + 1.1 * range_val * 1.1 / 4
        l3 = prev_close - 1.1 * range_val * 1.1 / 4
        l4 = prev_close - 1.1 * range_val * 1.1 / 2
        
        # Align to 12h: each 1d bar = 2x 12h bars
        start_idx = i * 2
        end_idx = start_idx + 2
        if end_idx <= n:
            camarilla_high[start_idx:end_idx] = h4
            camarilla_low[start_idx:end_idx] = l4
            camarilla_h3[start_idx:end_idx] = h3
            camarilla_l3[start_idx:end_idx] = l3
    
    # Align Camarilla levels to 12h (already done in loop above)
    # Get daily volume for confirmation (>2.0x 24-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates strong trending market
        strong_trend = adx_1w_aligned[i] > 25
        
        # Camarilla breakout conditions
        breakout_h3 = close[i] > camarilla_h3[i]
        breakout_l3 = close[i] < camarilla_l3[i]
        
        # Exit conditions: touch H4/L4 levels or retest H3/L3
        touch_h4 = close[i] > camarilla_high[i]  # Exit long on H4 touch
        touch_l4 = close[i] < camarilla_low[i]   # Exit short on L4 touch
        retest_h3 = close[i] < camarilla_h3[i] and position == 1  # Long exit on H3 retest
        retest_l3 = close[i] > camarilla_l3[i] and position == -1  # Short exit on L3 retest
        
        # Entry logic: Camarilla breakout + strong trend + volume confirmation
        long_entry = breakout_h3 and strong_trend and volume_spike[i]
        short_entry = breakout_l3 and strong_trend and volume_spike[i]
        
        # Exit logic: H4/L4 touch or H3/L3 retest
        long_exit = touch_h4 or retest_h3
        short_exit = touch_l4 or retest_l3
        
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

name = "12h_1d_camarilla_h3l3_breakout_adx_volume_v1"
timeframe = "12h"
leverage = 1.0