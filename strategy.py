#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with weekly trend filter and volume confirmation.
    # Long when weekly trend is up (price > weekly EMA50) AND Williams %R < -80 (oversold) AND 6h volume > 1.5x 20-period MA.
    # Short when weekly trend is down (price < weekly EMA50) AND Williams %R > -20 (overbought) AND 6h volume > 1.5x 20-period MA.
    # Exit when Williams %R crosses back to -50 (mean reversion midpoint).
    # Uses Williams %R for mean reversion signals, weekly EMA for trend filter, volume for confirmation.
    # Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - optional but good practice
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready or outside session (optional)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        volume_spike = volume_6h_aligned[i] > 1.5 * vol_ma_6h_aligned[i]
        
        # Trend filter: price relative to weekly EMA50
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        exit_signal = abs(williams_r_aligned[i] + 50) < 5  # Exit near -50 midpoint
        
        # Entry conditions
        if weekly_uptrend and oversold and volume_spike and position != 1:
            position = 1
            signals[i] = position_size
        elif weekly_downtrend and overbought and volume_spike and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif exit_signal and position != 0:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0