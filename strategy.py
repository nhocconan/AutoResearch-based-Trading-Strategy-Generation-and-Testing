#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian(20) breakout with 4h EMA20 trend filter and 1d ATR volume filter.
Long when price breaks above Donchian upper(20) AND 4h EMA20 rising AND 1d volume > 1.5x ATR(20).
Short when price breaks below Donchian lower(20) AND 4h EMA20 falling AND 1d volume > 1.5x ATR(20).
Exit when price touches opposite Donchian level or 4h EMA20 reverses.
Uses 4h HTF for trend, 1d for volume/volatility filter to reduce false breakouts in ranging markets.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Donchian channels provide structure, EMA20 filters trend, volume/ATR avoids low-momentum breakouts.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA20 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1h Donchian channels (20-period)
    lookback = 20
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_upper[i] = np.max(high[i-lookback+1:i+1])
        donchian_lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 1d ATR(20) for volume filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    # Calculate 1d volume average (20-period) for spike filter
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 20, 20)  # Donchian, EMA20, ATR/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr_20_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_20_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        atr_val = atr_20_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        
        # Calculate EMA20 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_20_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1d volume > 1.5x ATR(20) (adaptive to volatility)
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Break above Donchian upper AND EMA20 rising AND volume filter
            if price > upper and ema_rising and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below Donchian lower AND EMA20 falling AND volume filter
            elif price < lower and ema_falling and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower Donchian OR EMA20 starts falling
                if price < lower or (i >= start_idx + 1 and ema_val < ema_20_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper Donchian OR EMA20 starts rising
                if price > upper or (i >= start_idx + 1 and ema_val > ema_20_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Donchian20_Breakout_4hEMA20_Trend_1dATRVolFilter"
timeframe = "1h"
leverage = 1.0