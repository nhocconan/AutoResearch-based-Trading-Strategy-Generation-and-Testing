#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + daily volume confirmation + choppiness regime filter
# Long when price breaks above 12h Donchian upper band (20-period high), volume > 1.5x 20-period average, and choppy market (CHOP > 61.8)
# Short when price breaks below 12h Donchian lower band (20-period low), volume > 1.5x 20-period average, and choppy market (CHOP > 61.8)
# Uses daily timeframe for volume and choppiness to reduce noise and avoid whipsaws
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull and bear markets due to volatility-based breakouts and regime filtering

name = "12h_Donchian20_Volume_Chop"
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
    
    # Get daily data once for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily volume average (20-period)
    daily_volume = df_1d['volume'].values
    vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Calculate daily choppiness index (14-period)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(daily_high - daily_low)
    tr2 = np.abs(np.diff(daily_close, prepend=daily_close[0]))
    tr3 = np.abs(np.roll(daily_close, 1) - daily_close)
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Max High and Min Low over 14 periods
    max_high = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    
    # Sum of ATR over 14 periods
    atr_sum = np.zeros_like(daily_close)
    for i in range(len(daily_close)):
        start = max(0, i-14)
        atr_sum[i] = np.sum(atr[start:i+1]) if i >= start else 0
    
    # Choppiness Index
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14), 
                    50)
    chop = np.where((max_high - min_low) == 0, 50, chop)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ma_val = vol_ma_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume confirmation, choppy market
            if close[i] > donchian_high[i] and volume[i] > 1.5 * vol_ma_val and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume confirmation, choppy market
            elif close[i] < donchian_low[i] and volume[i] > 1.5 * vol_ma_val and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or choppy condition fails
            if close[i] < donchian_low[i] or chop_val <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or choppy condition fails
            if close[i] > donchian_high[i] or chop_val <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals