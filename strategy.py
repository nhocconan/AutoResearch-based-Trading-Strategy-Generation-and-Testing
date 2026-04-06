#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly ATR filter and volume confirmation.
# Uses 20-day Donchian channels for breakout entries, filtered by weekly ATR to avoid
# low volatility false breakouts and volume confirmation for institutional participation.
# Designed for 1d timeframe targeting 30-100 trades over 4 years with low frequency,
# suitable for both bull and bear markets via breakout logic.

name = "1d_donchian20_atr1v_vol_v1"
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
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # 1-week ATR for volatility filter (20-period)
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
    
    # ATR calculation with 20-period smoothing
    atr_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        atr_1w[19] = np.mean(tr[1:20])
        for i in range(20, len(close_1w)):
            atr_1w[i] = (atr_1w[i-1] * 19 + tr[i]) / 20
    
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 1-week volume average for confirmation (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(19, len(vol_1w)):
        vol_ma_1w[i] = np.mean(vol_1w[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 19)  # Donchian needs 19, ATR needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above average (avoid low volatility breakouts)
        vol_filter = atr_aligned[i] > np.nanmean(atr_aligned[:i]) * 0.8 if i > 20 else True
        
        # Volume condition: current volume > 1.2x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.2
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower band or stoploss
            if (close[i] < lowest_low[i] or 
                close[i] < entry_price - 2.5 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper band or stoploss
            if (close[i] > highest_high[i] or 
                close[i] > entry_price + 2.5 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume and volatility confirmation
            if volume_filter and vol_filter:
                # Long: price breaks above upper band
                if close[i] > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below lower band
                elif close[i] < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals