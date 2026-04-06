#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA trend filter and volume confirmation.
# Breakouts capture momentum in both bull and bear markets. EMA filter ensures trend alignment.
# Volume confirmation reduces false breakouts. Targets 100-200 trades over 4 years with low frequency.
# Stoploss at 2.5*ATR limits drawdown.

name = "4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4-hour Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # 12-hour EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * 49) / 51
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 1-day volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # ATR for volatility and stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(13, n):
        if i == 13:
            atr[i] = np.mean(tr[1:14])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 49, 19, 13)  # Donchian needs 19, EMA needs 49, volume needs 19, ATR needs 13
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Using previous bar's high to avoid look-ahead
        breakout_down = close[i] < lowest_low[i-1]   # Using previous bar's low
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: breakdown or stoploss
            if (breakout_down or 
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: breakout or stoploss
            if (breakout_up or 
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume and trend confirmation
            if volume_filter:
                # Long: upward breakout in uptrend
                if breakout_up and uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: downward breakout in downtrend
                elif breakout_down and downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals