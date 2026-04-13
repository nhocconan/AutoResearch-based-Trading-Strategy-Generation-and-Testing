#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h Camarilla H3/L3 breakout with 1d ADX regime filter
    # Long: price breaks above 4h H3 AND 1d ADX > 25 (trending) AND session filter (08-20 UTC)
    # Short: price breaks below 4h L3 AND 1d ADX > 25 (trending) AND session filter (08-20 UTC)
    # Exit: price touches 4h H4/L4 OR retests opposite H3/L3 level
    # Using 4h/1d for signal direction, 1h only for entry timing to avoid overtrading.
    # Target: 15-37 trades/year on 1h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
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
    
    # Align daily ADX to 1h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Previous day's OHLC for 4h Camarilla calculation
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Camarilla levels: H3, L3, H4, L4
    camarilla_h3 = prev_close + 1.25 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.25 * (prev_high - prev_low)
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1h
    h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        trending_market = adx_1d_aligned[i] > 25
        
        # Camarilla breakout conditions
        breakout_h3 = close[i] > h3_aligned[i]
        breakout_l3 = close[i] < l3_aligned[i]
        
        # Exit conditions: touch H4/L4 or retest opposite level
        touch_h4 = close[i] >= h4_aligned[i]
        touch_l4 = close[i] <= l4_aligned[i]
        retest_h3 = close[i] < h3_aligned[i] and position == 1  # Long exit on H3 retest
        retest_l3 = close[i] > l3_aligned[i] and position == -1  # Short exit on L3 retest
        
        # Entry logic: Camarilla breakout + trending market + session filter
        long_entry = breakout_h3 and trending_market
        short_entry = breakout_l3 and trending_market
        
        # Exit logic: H4/L4 touch or opposite level retest
        long_exit = touch_h4 or retest_h3
        short_exit = touch_l4 or retest_l3
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_h3l3_breakout_adx_session_v1"
timeframe = "1h"
leverage = 1.0