#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with 1-day EMA trend filter and 1-week volume confirmation.
# Donchian breakouts capture trend continuation; EMA filter ensures alignment with higher timeframe trend.
# Volume confirmation filters weak breakouts. Designed for low frequency (target: 50-150 trades over 4 years).
# Works in bull markets via breakout longs and bear markets via breakout shorts.

name = "12h_donchian20_1d_ema20_vol_v1"
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
    
    # 1-day EMA(20) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        ema_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 19) / 21
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    if len(vol_1w) >= 5:
        for i in range(4, len(vol_1w)):
            vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 4)  # EMA needs 20, volume needs 4
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x weekly average
        volume_filter = volume[i] > vol_ma_1w_aligned[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below lower Donchian(20) or stoploss
            if (i >= 20 and low[i] <= np.min(low[i-20:i]) or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above upper Donchian(20) or stoploss
            if (i >= 20 and high[i] >= np.max(high[i-20:i]) or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend and volume
            if i >= 20 and volume_filter:
                upper_dc = np.max(high[i-20:i])
                lower_dc = np.min(low[i-20:i])
                
                # Long: price breaks above upper Donchian in uptrend
                if high[i] > upper_dc and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below lower Donchian in downtrend
                elif low[i] < lower_dc and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals