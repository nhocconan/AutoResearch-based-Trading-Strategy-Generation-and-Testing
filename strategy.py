#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# In trending markets (1d CHOP < 42): breakout in direction of 1d EMA34 trend
# In ranging markets (1d CHOP >= 42): fade Camarilla touches (mean reversion)
# Volume confirmation (>1.5x 20-period EMA) filters low-quality breakouts
# Discrete sizing (0.25) minimizes fee churn. Target: 75-200 trades over 4 years.
# Strategy adapts to bull/bear markets via regime filter and uses 4h primary timeframe.

name = "4h_Camarilla_R3S3_1dChop_Volume_EMA34"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filter and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d Choppiness Index (CHOP) - 14 period
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d.sub(low_1d)
    tr2 = high_1d.sub(close_1d.shift(1)).abs()
    tr3 = low_1d.sub(close_1d.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over 14 periods
    tr_sum_14 = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh_14 = high_1d.rolling(window=14, min_periods=14).max()
    ll_14 = low_1d.rolling(window=14, min_periods=14).min()
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hh_ll_diff = hh_14 - ll_14
    chop_1d = np.where(
        (hh_ll_diff > 0) & (~tr_sum_14.isna()) & (~hh_ll_diff.isna()),
        100 * np.log10(tr_sum_14 / hh_ll_diff) / np.log10(14),
        50.0  # neutral when undefined
    )
    
    # Align 1d indicators to 4h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Camarilla levels (R3, S3, R4, S4) from previous day
    # Camarilla levels based on previous day's OHLC
    # We need to shift by 1 to avoid look-ahead (use previous completed day)
    # For 4h bars, we calculate levels from the 1d data and align
    
    # Previous day's OHLC (1d data shifted by 1)
    prev_close_1d = close_1d.shift(1)
    prev_high_1d = high_1d.shift(1)
    prev_low_1d = low_1d.shift(1)
    prev_open_1d = df_1d['open'].shift(1).values  # assuming open column exists
    
    # Typical Camarilla calculation using previous day's range
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    # But we'll use the more common formula based on close and range
    # Actually, standard Camarilla uses:
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    
    # Calculate using previous day's data
    prev_range = prev_high_1d - prev_low_1d
    camarilla_r4 = prev_close_1d + 1.5 * prev_range
    camarilla_r3 = prev_close_1d + 1.0 * prev_range
    camarilla_s3 = prev_close_1d - 1.0 * prev_range
    camarilla_s4 = prev_close_1d - 1.5 * prev_range
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            if chop_aligned[i] < 42:
                # Trending market: breakout in direction of 1d EMA34 trend
                # Determine trend: price above/below EMA34
                if close[i] > ema34_aligned[i]:
                    # Uptrend: long on break above R3
                    if close[i] > r3_aligned[i] and volume_confirm:
                        signals[i] = 0.25
                        position = 1
                else:
                    # Downtrend: short on break below S3
                    if close[i] < s3_aligned[i] and volume_confirm:
                        signals[i] = -0.25
                        position = -1
            else:
                # Ranging market: fade Camarilla touches (mean reversion)
                if close[i] <= s3_aligned[i] and volume_confirm:
                    # Long at S3
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= r3_aligned[i] and volume_confirm:
                    # Short at R3
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to midpoint between R3 and S3 OR chop increases (>50) OR volume drops
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] >= midpoint or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint OR chop increases (>50) OR volume drops
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] <= midpoint or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals