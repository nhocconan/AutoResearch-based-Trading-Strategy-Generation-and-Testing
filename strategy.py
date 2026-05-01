#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# Uses 1w EMA50 to define trend: price > EMA50 = uptrend, price < EMA50 = downtrend
# Entry: Long when price breaks above Donchian(20) high AND volume > 1.5 * avg_volume(20) in uptrend
# Entry: Short when price breaks below Donchian(20) low AND volume > 1.5 * avg_volume(20) in downtrend
# Exit: Opposite Donchian breakout or regime change (price crosses EMA50)
# Designed for low frequency (50-150 trades over 4 years) with clear trend following logic
# Works in bull markets via breakouts and in bear markets via short breakdowns

name = "12h_Donchian20_1wEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian(20) channels
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma20[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20, 19)  # Need EMA50, Donchian, volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Uptrend: look for long breakouts
            if uptrend:
                if high[i] > highest_high[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            # Downtrend: look for short breakdowns
            elif downtrend:
                if low[i] < lowest_low[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Price exactly at EMA50 - stay flat
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on Donchian breakdown (opposite signal)
            if low[i] < lowest_low[i]:
                exit_long = True
            # Exit on trend change (price crosses below EMA50)
            elif close[i] <= ema50_1w_aligned[i]:
                exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on Donchian breakout (opposite signal)
            if high[i] > highest_high[i]:
                exit_short = True
            # Exit on trend change (price crosses above EMA50)
            elif close[i] >= ema50_1w_aligned[i]:
                exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals