#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume filter: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly OHLC for Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's Camarilla pivot levels (to avoid look-ahead)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r3_1w = close_1w + range_1w * 1.166
    s3_1w = close_1w - range_1w * 1.166
    
    # Shift by 1 to use only completed weekly bars
    r3_1w = np.roll(r3_1w, 1)
    s3_1w = np.roll(s3_1w, 1)
    r3_1w[0] = np.nan
    s3_1w[0] = np.nan
    
    # Align weekly Camarilla levels to daily timeframe
    r3_1d = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1d = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Session filter: 0-23 UTC (daily bars cover full day)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    # Minimum holding period: 2 days to reduce churn
    hold_count = np.zeros(n, dtype=int)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Decrease hold counter
        if hold_count[i] > 0:
            hold_count[i] -= 1
        
        # Skip if any required data is invalid or outside session or holding
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i] or hold_count[i] > 0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: daily volume must be elevated
        volume_confirmed = volume_current > 2.0 * vol_ma
        
        # Long conditions: price breaks above weekly R3 with volume
        long_signal = volume_confirmed and (price_high > r3_1d[i])
        
        # Short conditions: price breaks below weekly S3 with volume
        short_signal = volume_confirmed and (price_low < s3_1d[i])
        
        # Exit when price returns to the weekly pivot (mean reversion)
        pivot_1w_today = (high_1w + low_1w + close_1w) / 3
        pivot_1d = align_htf_to_ltf(prices, df_1w, pivot_1w_today)
        exit_long = position == 1 and price_close < pivot_1d[i]
        exit_short = position == -1 and price_close > pivot_1d[i]
        
        # Trading logic with minimum holding period
        if long_signal and position != 1:
            position = 1
            hold_count[i] = 2  # Hold for 2 days minimum
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            hold_count[i] = 2  # Hold for 2 days minimum
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

# Hypothesis: 1d Camarilla breakout using weekly pivot levels with volume confirmation.
# Uses weekly Camarilla R3/S3 levels (previous week's high/low/close) for weekly structure.
# Enters long when daily price breaks above weekly R3 with volume >2.0x daily 20-period average.
# Enters short when daily price breaks below weekly S3 with same volume conditions.
# Exits when price returns to the weekly pivot level (mean reversion within the weekly range).
# Weekly timeframe reduces trade frequency to target 30-100 trades over 4 years (7-25/year).
# Works in both bull and bear markets by capturing breakouts with volume confirmation.