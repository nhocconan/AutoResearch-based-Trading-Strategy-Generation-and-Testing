#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(15) breakout with 1d ATR-based volatility filter and 1w EMA trend filter.
    # Donchian breakouts capture momentum. ATR filter ensures breakouts occur during sufficient volatility.
    # 1w EMA (50) provides multi-week trend bias: only long when price > weekly EMA, short when price < weekly EMA.
    # This avoids counter-trend breakouts in strong trends. Target: 50-150 total trades over 4 years.
    # Works in bull markets (long breakouts with bullish weekly trend) and bear markets (short breakouts with bearish weekly trend).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:15])  # first ATR
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align 1d ATR to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian channels (15-period)
    donchian_high = pd.Series(high).rolling(window=15, min_periods=15).max().values
    donchian_low = pd.Series(low).rolling(window=15, min_periods=15).min().values
    
    # Calculate 12h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 20-period MA
        volume_filter = volume[i] > volume_ma[i]
        
        # ATR filter: current ATR > 0.5 * 20-period MA ATR (ensures sufficient volatility)
        atr_ma = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean().values
        atr_filter = atr_1d_aligned[i] > 0.5 * atr_ma[i]
        
        # Weekly trend filter: price above/below weekly EMA
        bullish_trend = close[i] > ema_50_1w_aligned[i]
        bearish_trend = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above prior period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below prior period's low
        
        # Entry conditions: breakout in direction of weekly trend with volume and ATR confirmation
        long_entry = long_breakout and bullish_trend and volume_filter and atr_filter
        short_entry = short_breakout and bearish_trend and volume_filter and atr_filter
        
        # Exit conditions: opposite breakout or loss of trend
        long_exit = short_breakout or not bullish_trend
        short_exit = long_breakout or not bearish_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_donchian_atr_ema_v1"
timeframe = "12h"
leverage = 1.0