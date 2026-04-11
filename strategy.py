#!/usr/bin/env python3
"""
1d_1w_camarilla_breakout_volume_trend_v1
Strategy: 1d breakout with 1w trend filter and volume confirmation
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses weekly trend (1w EMA) to filter daily Camarilla breakouts, reducing false signals in chop. Volume confirmation ensures breakout strength. Designed for fewer trades (target 10-20/year) to avoid fee drag while capturing strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1w EMA (trend filter) ===
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === 1d Camarilla (entry levels from prior 1d) ===
    high_1d = high
    low_1d = low
    close_1d = close
    
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    pivot_1d = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_1d = high_1d_shift - low_1d_shift
    r3_1d = close_1d_shift + range_1d * 1.166
    s3_1d = close_1d_shift - range_1d * 1.166
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_open = open_price[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 1d volume must be expanded
        volume_expanded = volume_current > 1.5 * vol_ma
        
        # Trend filter: price above/below 1w EMA
        above_ema = price_close > ema_1w_aligned[i]
        below_ema = price_close < ema_1w_aligned[i]
        
        # Long conditions: 1d close above prior day's R3 with volume expansion + above weekly EMA
        long_signal = volume_expanded and above_ema and (price_close > r3_1d[i])
        
        # Short conditions: 1d close below prior day's S3 with volume expansion + below weekly EMA
        short_signal = volume_expanded and below_ema and (price_close < s3_1d[i])
        
        # Exit when price returns to the 1d pivot (mean reversion within prior day's range)
        exit_long = position == 1 and price_close < pivot_1d[i]
        exit_short = position == -1 and price_close > pivot_1d[i]
        
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

# Hypothesis: Uses weekly trend (1w EMA) to filter daily Camarilla breakouts, reducing false signals in chop. Volume confirmation ensures breakout strength. Designed for fewer trades (target 10-20/year) to avoid fee drag while capturing strong trends in both bull and bear markets.