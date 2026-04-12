#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Only trade breakouts in direction of weekly trend to avoid counter-trend whipsaws
# Volume filter ensures breakouts have conviction
# Target: 20-50 trades/year per symbol (75-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 1d timeframe (no shift needed as we use completed daily bars)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate 1d ATR(14) for volatility filter and position sizing
    tr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i == 0:
            tr_1d[i] = high_1d[i] - low_1d[i]
        else:
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    atr14_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if i == 13:
            atr14_1d[i] = np.mean(tr_1d[i-13:i+1])
        else:
            atr14_1d[i] = (atr14_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend direction
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 1d volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_20_aligned[i]
        short_breakout = close[i] < lowest_20_aligned[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Entry logic
        if long_breakout and weekly_uptrend and volume_confirm and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_breakout and weekly_downtrend and volume_confirm and position != -1:
            position = -1
            signals[i] = -0.30
        # Exit logic: reverse signal or loss of trend
        elif position == 1 and (short_breakout or not weekly_uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (long_breakout or not weekly_downtrend):
            position = 0
            signals[i] = 0.0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_breakout_trend_volume_v1"
timeframe = "1d"
leverage = 1.0