#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with 1-week ATR filter and volume confirmation.
# Buy when price breaks above 20-day high with expanding volume and low volatility (ATR < 20-day SMA).
# Sell when price breaks below 20-day low or volatility expands (ATR > 1.5x 20-day SMA).
# Designed for 1d timeframe to target 30-100 trades over 4 years with low frequency.
# Works in bull markets (breakouts) and bear markets (breakdowns) with volatility filter.

name = "1d_donchian20_vol_atr_v1"
timeframe = "1d"
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
    
    # 1-day Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):  # 20-period lookback
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # 1-week ATR for volatility filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr = np.full(len(close_1w), np.nan)
    if len(close_1w) > 1:
        tr[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr[i] = max(high_1w[i] - low_1w[i],
                       abs(high_1w[i] - close_1w[i-1]),
                       abs(low_1w[i] - close_1w[i-1]))
    
    # ATR(14) calculation
    atr_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 14:
        atr_1w[13] = np.mean(tr[1:15])  # First ATR at index 13
        for i in range(14, len(close_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to daily timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 20-day SMA of ATR for volatility regime
    atr_sma = np.full(n, np.nan)
    for i in range(33, n):  # Need 14 (ATR) + 19 (SMA lookback) = 33
        if not np.isnan(atr_1w_aligned[i-19:i+1]).any():
            atr_sma[i] = np.mean(atr_1w_aligned[i-19:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_sma = np.full(n, np.nan)
    for i in range(19, n):
        vol_sma[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > vol_sma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 33  # ATR(14) + 20-day SMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(atr_sma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: low volatility environment (ATR < SMA)
        low_vol = atr_1w_aligned[i] < atr_sma[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: breakdown below 20-day low or high volatility
            if (close[i] < lowest_low[i] or 
                atr_1w_aligned[i] > 1.5 * atr_sma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: breakout above 20-day high or high volatility
            if (close[i] > highest_high[i] or 
                atr_1w_aligned[i] > 1.5 * atr_sma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakouts in low volatility
            if low_vol and volume_filter[i]:
                # Long: breakout above 20-day high
                if close[i] > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below 20-day low
                elif close[i] < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals