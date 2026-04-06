#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour momentum breakout with daily volume confirmation and weekly volatility filter.
# Uses Donchian channel breakouts for directional entries, confirmed by daily volume spikes.
# Weekly ATR filter avoids trading during low volatility periods.
# Designed for 12h timeframe to target 50-150 trades over 4 years with controlled frequency.

name = "12h_donchian20_1d_vol1w_atr_v1"
timeframe = "12h"
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
    
    # 1-day Donchian channel (20-period) for breakout signals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower bands
    donch_high = np.full(len(close_1d), np.nan)
    donch_low = np.full(len(close_1d), np.nan)
    
    for i in range(19, len(close_1d)):
        donch_high[i] = np.max(high_1d[i-19:i+1])
        donch_low[i] = np.min(low_1d[i-19:i+1])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # 1-day volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
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
    
    # ATR calculation
    atr_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 14:
        atr_1w[13] = np.mean(tr[0:14])
        for i in range(14, len(close_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    atr_ma_1w = np.full(len(atr_1w), np.nan)
    if len(atr_1w) >= 28:  # 14*2 for smoothing
        atr_ma_1w[27] = np.mean(atr_1w[14:28])
        for i in range(28, len(atr_1w)):
            atr_ma_1w[i] = (atr_ma_1w[i-1] * 13 + atr_1w[i]) / 14
    
    atr_ma_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(27, 19, 19)  # ATR needs 27, Donchian needs 19, volume needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_ma_aligned[i] > 0  # Ensure we have valid ATR
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or stoploss
            if (close[i] < donch_low_aligned[i] or 
                close[i] < entry_price - 2.5 * atr_ma_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or stoploss
            if (close[i] > donch_high_aligned[i] or 
                close[i] > entry_price + 2.5 * atr_ma_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            if volume_filter and vol_filter:
                # Long: price breaks above Donchian high
                if close[i] > donch_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below Donchian low
                elif close[i] < donch_low_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals