#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_volume_v8"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h OHLC for Camarilla pivot levels (previous 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 1d OHLC for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation (previous 4h bar)
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    # Resistance and support levels (previous 4h bar's data)
    r3_4h = close_4h + range_4h * 1.166
    s3_4h = close_4h - range_4h * 1.166
    
    # Shift by 1 to use only completed 4h bars (previous 4h bar's levels)
    r3_4h = np.roll(r3_4h, 1)
    s3_4h = np.roll(s3_4h, 1)
    r3_4h[0] = np.nan
    s3_4h[0] = np.nan
    
    # Align 4h Camarilla levels to 1h timeframe
    r3_1h = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_1h = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # 1d trend filter: 20-period EMA on daily close
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 1h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1h[i]) or np.isnan(s3_1h[i]) or
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 1h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: price breaks above R3 with volume and price above daily EMA20
        long_signal = volume_confirmed and (price_high > r3_1h[i]) and (price_close > ema_20_1d_aligned[i])
        
        # Short conditions: price breaks below S3 with volume and price below daily EMA20
        short_signal = volume_confirmed and (price_low < s3_1h[i]) and (price_close < ema_20_1d_aligned[i])
        
        # Exit when price returns to the 4h pivot (mean reversion)
        pivot_4h = (high_4h + low_4h + close_4h) / 3
        pivot_1h = align_htf_to_ltf(prices, df_4h, pivot_4h)
        exit_long = position == 1 and price_close < pivot_1h[i]
        exit_short = position == -1 and price_close > pivot_1h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1h Camarilla breakout using 4h pivot levels with volume confirmation and daily trend filter.
# Uses 4h Camarilla R3/S3 levels (previous 4h bar's high/low/close) for intraday structure.
# Enters long when 1h price breaks above 4h R3 with volume >1.5x 1h 20-period average and price above daily EMA20.
# Enters short when 1h price breaks below 4h S3 with same volume conditions and price below daily EMA20.
# Exits when price returns to the 4h pivot level (mean reversion within the 4h range).
# Daily EMA20 filter ensures trades align with higher timeframe trend, reducing false signals in ranging markets.
# Volume filter reduces false breakouts, targeting 15-37 trades per year per symbol.
# Position size: 0.20 to manage risk in volatile markets.
# Session filter (8-20 UTC) reduces noise trades outside active hours.
# Designed to work in both bull and bear markets by combining intraday breakouts with higher timeframe trend.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.