#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily OHLC for Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla pivot levels (to avoid look-ahead)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.166
    s3_1d = close_1d - range_1d * 1.166
    
    # Shift by 1 to use only completed daily bars
    r3_1d = np.roll(r3_1d, 1)
    s3_1d = np.roll(s3_1d, 1)
    r3_1d[0] = np.nan
    s3_1d[0] = np.nan
    
    # Align daily Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Session filter: 0-23 UTC (12h bars cover full day)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    # Minimum holding period: 2 bars (24 hours) to reduce churn
    hold_count = np.zeros(n, dtype=int)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Decrease hold counter
        if hold_count[i] > 0:
            hold_count[i] -= 1
        
        # Skip if any required data is invalid or outside session or holding
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i] or hold_count[i] > 0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 12h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: price breaks above R3 with volume
        long_signal = volume_confirmed and (price_high > r3_12h[i])
        
        # Short conditions: price breaks below S3 with volume
        short_signal = volume_confirmed and (price_low < s3_12h[i])
        
        # Exit when price returns to the daily pivot (mean reversion)
        pivot_1d_today = (high_1d + low_1d + close_1d) / 3
        pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d_today)
        exit_long = position == 1 and price_close < pivot_12h[i]
        exit_short = position == -1 and price_close > pivot_12h[i]
        
        # Trading logic with minimum holding period
        if long_signal and position != 1:
            position = 1
            hold_count[i] = 2  # Hold for 2 bars minimum
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            hold_count[i] = 2  # Hold for 2 bars minimum
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h Camarilla breakout using daily pivot levels with volume confirmation.
# Uses daily Camarilla R3/S3 levels (previous day's high/low/close) for intraday structure.
# Enters long when 12h price breaks above daily R3 with volume >1.5x 12h 20-period average.
# Enters short when 12h price breaks below daily S3 with same volume conditions.
# Exits when price returns to the daily pivot level (mean reversion within the daily range).
# Volume filter reduces false breakouts, targeting 15-30 trades per year per symbol.
# Minimum holding period of 2 bars (24 hours) reduces churn and forces trades to develop.
# Position size: 0.25 to manage risk in volatile markets.
# Works in both bull and bear markets by capturing breakouts with volume confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.