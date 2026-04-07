#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot mean reversion with 12-hour trend filter and volume spike confirmation
# Long when price touches S3 level, 12h ADX < 25 (low trend), and volume > 1.5x 20-period average
# Short when price touches R3 level, 12h ADX < 25, and volume > 1.5x 20-period average
# Exit when price crosses opposite pivot level (S1 for long, R1 for short) or closes beyond entry point
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 100-200 total trades over 4 years (25-50/year)

name = "6h_camarilla_pivot_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour data for pivot points and ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous day's range (using previous 12h bar)
    prev_close = np.roll(close_12h, 1)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close[0] = close_12h[0]
    prev_high[0] = high_12h[0]
    prev_low[0] = low_12h[0]
    
    # Camarilla pivot calculations
    range_12h = prev_high - prev_low
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Resistance levels
    r1 = camarilla_pivot + (range_12h * 1.0833 / 12)
    r2 = camarilla_pivot + (range_12h * 1.1666 / 6)
    r3 = camarilla_pivot + (range_12h * 1.25 / 4)
    r4 = camarilla_pivot + (range_12h * 1.5 / 2)
    
    # Support levels
    s1 = camarilla_pivot - (range_12h * 1.0833 / 12)
    s2 = camarilla_pivot - (range_12h * 1.1666 / 6)
    s3 = camarilla_pivot - (range_12h * 1.25 / 4)
    s4 = camarilla_pivot - (range_12h * 1.5 / 2)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Calculate 12-hour ADX (14-period) for trend filter
    high_12h_adx = df_12h['high'].values
    low_12h_adx = df_12h['low'].values
    close_12h_adx = df_12h['close'].values
    
    # True Range
    tr1 = high_12h_adx - low_12h_adx
    tr2 = np.abs(high_12h_adx - np.roll(close_12h_adx, 1))
    tr3 = np.abs(low_12h_adx - np.roll(close_12h_adx, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_12h_adx, prepend=high_12h_adx[0])
    down_move = np.diff(low_12h_adx, prepend=low_12h_adx[0]) * -1
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume spike confirmation (20-period average)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above S1 (take profit) or closes below entry
            elif close[i] > s1_aligned[i] or close[i] < entry_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below R1 (take profit) or closes above entry
            elif close[i] < r1_aligned[i] or close[i] > entry_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla S3/R3 touch with low trend and volume spike
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma[i]
            # Trend filter: 12h ADX < 25 (low trend environment for mean reversion)
            trend_filter = adx_aligned[i] < 25
            
            # Long: price touches or goes below S3 level + volume filter + trend filter
            if close[i] <= s3_aligned[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches or goes above R3 level + volume filter + trend filter
            elif close[i] >= r3_aligned[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals