#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h/1d trend filters. Use 4h EMA(50) for direction, 1d EMA(34) for regime filter,
# and 1h Donchian(20) breakout for entry. Only trade in direction of higher timeframe trends.
# Volume confirmation (>1.5x 20-period average) reduces false breakouts.
# Session filter (08-20 UTC) avoids low-liquidity periods.
# Target: 15-30 trades/year by requiring 4h/1d alignment + breakout + volume + session.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA(34) for regime filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h Donchian(20) channels
    high_max = prices['high'].rolling(window=20, min_periods=20).max().values
    low_min = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after EMA warmup
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filters: price above/below HTF EMAs
        above_4h = price > ema_4h_aligned[i]
        above_1d = price > ema_1d_aligned[i]
        
        if position == 0:
            # Look for breakout in direction of higher timeframe trends
            if above_4h and above_1d and volume_confirm:
                # Bullish alignment: long on breakout above Donchian high
                if price > high_max[i]:
                    signals[i] = 0.20
                    position = 1
            elif not above_4h and not above_1d and volume_confirm:
                # Bearish alignment: short on breakdown below Donchian low
                if price < low_min[i]:
                    signals[i] = -0.20
                    position = -1
        
        elif position != 0:
            # Exit when price crosses opposite HTF EMA or loses volume confirmation
            exit_signal = False
            
            if position == 1:  # long position
                if price < ema_4h_aligned[i] or price < ema_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:  # short position
                if price > ema_4h_aligned[i] or price > ema_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_4h1d_EMA_Trend_Donchian_Breakout_Volume_Session"
timeframe = "1h"
leverage = 1.0