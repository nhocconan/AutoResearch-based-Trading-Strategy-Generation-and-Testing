#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (EMA34) and volume confirmation (volume > 1.5x weekly VWAP)
# Long when price breaks above upper Donchian + price > weekly EMA34 + volume > 1.5x weekly VWAP
# Short when price breaks below lower Donchian + price < weekly EMA34 + volume > 1.5x weekly VWAP
# Exit when price returns to Donchian midline (midpoint of upper/lower)
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Works in both bull and bear: trend filter captures direction, volume confirms strength, Donchian breakout captures momentum

name = "1d_Donchian_20_1wEMA34_VolumeVWAP"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate weekly VWAP for volume filter (approximated as typical price * volume)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap = (typical_price * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap = vwap.values
    
    # Align weekly indicators to 1d timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    
    # Calculate daily Donchian channels (20-period)
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    # Middle = (Upper + Lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    middle = (upper + lower) / 2
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_34_aligned[i]) or np.isnan(vwap_aligned[i]) or \
           np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current weekly volume > 1.5x weekly VWAP (as proxy for institutional interest)
        # Find the most recent completed weekly bar
        idx_1w = 0
        while idx_1w < len(df_1w) and df_1w.iloc[idx_1w]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1w += 1
        idx_1w -= 1  # last completed weekly bar
        
        if idx_1w < 0:
            vol_filter = False
        else:
            vol_1w_current = df_1w.iloc[idx_1w]['volume']
            # Compare to 20-period average volume for normalization
            vol_ma_20 = df_1w['volume'].rolling(window=20, min_periods=20).mean().iloc[idx_1w]
            vol_filter = vol_1w_current > 1.5 * vol_ma_20 if not pd.isna(vol_ma_20) else False
        
        if position == 0:
            # Look for entry: Donchian breakout + trend + volume
            long_condition = close[i] > upper[i] and close[i] > ema_34_aligned[i] and vol_filter
            short_condition = close[i] < lower[i] and close[i] < ema_34_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midline
            if close[i] <= middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midline
            if close[i] >= middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals