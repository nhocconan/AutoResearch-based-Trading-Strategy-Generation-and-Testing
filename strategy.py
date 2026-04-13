#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H3/L3 breakout with 1d ADX regime filter and volume confirmation
    # Long: price breaks above H3 AND ADX(14) > 25 (trending) AND volume > 1.5x avg
    # Short: price breaks below L3 AND ADX(14) > 25 (trending) AND volume > 1.5x avg
    # Exit: price touches H4 (for longs) or L4 (for shorts) OR opposite Camarilla level touch
    # Using 12h timeframe for optimal trade frequency (target 12-37/year), Camarilla levels for intraday structure,
    # ADX to filter ranging markets, and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX regime filter
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
    
    # Smoothed values (Wilder's smoothing)
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
    
    # Align daily ADX to 12h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    # We need to get daily OHLC and align it to 12h bars
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d_ohlc['close'].shift(1).values
    prev_high = df_1d_ohlc['high'].shift(1).values
    prev_low = df_1d_ohlc['low'].shift(1).values
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_h3 = prev_close + 1.25 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.25 * (prev_high - prev_low)
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h
    h3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_l4)
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        trending_market = adx_1d_aligned[i] > 25
        
        # Camarilla breakout conditions
        breakout_h3 = close[i] > h3_aligned[i]
        breakout_l3 = close[i] < l3_aligned[i]
        
        # Exit conditions: touch H4/L4 or opposite level
        touch_h4 = close[i] >= h4_aligned[i]
        touch_l4 = close[i] <= l4_aligned[i]
        touch_opposite_h3 = close[i] < h3_aligned[i] and position == 1  # Long exit on H3 retest
        touch_opposite_l3 = close[i] > l3_aligned[i] and position == -1  # Short exit on L3 retest
        
        # Entry logic: Camarilla breakout + trending market + volume confirmation
        long_entry = breakout_h3 and trending_market and volume_spike[i]
        short_entry = breakout_l3 and trending_market and volume_spike[i]
        
        # Exit logic: H4/L4 touch or opposite level retest
        long_exit = touch_h4 or touch_opposite_l3
        short_exit = touch_l4 or touch_opposite_h3
        
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