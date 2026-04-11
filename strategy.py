#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivot levels (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly OHLC for additional filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
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
    
    # Weekly trend filter: 20-period EMA on weekly close
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 4h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Additional: Daily volume ratio filter (volume > 2x daily average)
    # Use daily volume from df_1d and align to 4h
    vol_1d = df_1d['volume'].values
    vol_ma_10_1d = pd.Series(vol_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_10_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(vol_ma_10_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        vol_daily_ma = vol_ma_10_1d_aligned[i]
        
        # Volume confirmation: both 4h and daily volume must be elevated
        volume_confirmed = (volume_current > 1.5 * vol_ma) and (volume_current > 2.0 * vol_daily_ma)
        
        # Long conditions: price breaks above R3 with volume and price above weekly EMA20
        long_signal = volume_confirmed and (price_high > r3_4h[i]) and (price_close > ema_20_1w_aligned[i])
        
        # Short conditions: price breaks below S3 with volume and price below weekly EMA20
        short_signal = volume_confirmed and (price_low < s3_4h[i]) and (price_close < ema_20_1w_aligned[i])
        
        # Exit when price returns to the daily pivot (mean reversion)
        pivot_1d = (high_1d + low_1d + close_1d) / 3
        pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
        exit_long = position == 1 and price_close < pivot_4h[i]
        exit_short = position == -1 and price_close > pivot_4h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
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

# Hypothesis: 4h Camarilla breakout using daily pivot levels with dual volume confirmation (4h and daily) and weekly trend filter.
# Uses daily Camarilla R3/S3 levels (previous day's high/low/close) for intraday structure.
# Enters long when 4h price breaks above daily R3 with volume >1.5x 4h 20-period average AND >2x daily 10-period average
# and price above weekly EMA20. Enters short when 4h price breaks below daily S3 with same volume conditions
# and price below weekly EMA20. Exits when price returns to the daily pivot level (mean reversion within the day's range).
# Weekly EMA20 filter ensures trades align with higher timeframe trend, reducing false signals in ranging markets.
# Dual volume filter significantly reduces false breakouts, targeting only 20-40 trades per year per symbol.
# Position size: 0.25 to balance risk and return, limiting drawdown in volatile markets.
# Designed to work in both bull and bear markets by combining intraday breakouts with higher timeframe trend.
# Target: 50-100 trades over 4 years (12-25/year) to minimize fee drag.