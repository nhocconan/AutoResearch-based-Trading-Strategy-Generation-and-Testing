#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_camarilla_breakout_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivot levels (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation (previous day)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance and support levels (previous day's data)
    r3_1d = close_1d + range_1d * 1.166
    s3_1d = close_1d - range_1d * 1.166
    
    # Shift by 1 to use only completed daily bars (previous day's levels)
    r3_1d = np.roll(r3_1d, 1)
    s3_1d = np.roll(s3_1d, 1)
    r3_1d[0] = np.nan
    s3_1d[0] = np.nan
    
    # Align daily Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate weekly ATR for volatility filter (14 period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ATR to 4h timeframe
    atr_4h = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or
            np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: price breaks above R3 with volume
        long_signal = volume_confirmed and (price_high > r3_4h[i])
        
        # Short conditions: price breaks below S3 with volume
        short_signal = volume_confirmed and (price_low < s3_4h[i])
        
        # Exit when price returns to the daily pivot (mean reversion)
        pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
        exit_long = position == 1 and price_close < pivot_4h[i]
        exit_short = position == -1 and price_close > pivot_4h[i]
        
        # ATR-based stop loss (2x weekly ATR)
        stop_long = position == 1 and price_close < (price_high - 2 * atr_4h[i])
        stop_short = position == -1 and price_close > (price_low + 2 * atr_4h[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h Camarilla breakout strategy using daily pivot levels with weekly volatility filter.
# Uses daily Camarilla R3/S3 levels (previous day's high/low/close) for intraday structure.
# Enters long when 4h price breaks above daily R3 with volume >1.5x 20-period average.
# Enters short when 4h price breaks below daily S3 with volume >1.5x 20-period average.
# Exits when price returns to the daily pivot level (mean reversion within the day's range).
# Uses weekly ATR for volatility-adjusted stop loss to adapt to changing market conditions.
# Weekly timeframe provides volatility context without adding look-ahead bias.
# Volume confirmation filters out low-conviction breakouts.
# Position size: 0.25 to balance risk and return, limiting drawdown in volatile markets.
# Designed to work in both bull and bear markets by adapting to daily volatility ranges.
# Target: 50-150 trades over 4 years (12-38/year) to minimize fee drag.