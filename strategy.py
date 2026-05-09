#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA(50) trend filter + volume confirmation
# Uses weekly EMA for trend direction, daily Donchian breakout for entries, and volume filter to avoid false breakouts.
# Designed to capture trend continuations while avoiding counter-trend trades in ranging markets. Target: 20-40 trades/year.
name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 1d high/low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high over last 20 periods
    upper_1d = np.full(len(high_1d), np.nan)
    for i in range(20, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-20:i])
    
    # Lower band: lowest low over last 20 periods
    lower_1d = np.full(len(low_1d), np.nan)
    for i in range(20, len(low_1d)):
        lower_1d[i] = np.min(low_1d[i-20:i])
    
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period volume moving average on 1d
    vol_ma_20 = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        upper = upper_1d_aligned[i]
        lower = lower_1d_aligned[i]
        ema = ema_50_1w_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian AND price > weekly EMA50 AND volume > 20-period average
            if price > upper and price > ema and vol > vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian AND price < weekly EMA50 AND volume > 20-period average
            elif price < lower and price < ema and vol > vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR price < weekly EMA50
            if price < lower or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR price > weekly EMA50
            if price > upper or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals