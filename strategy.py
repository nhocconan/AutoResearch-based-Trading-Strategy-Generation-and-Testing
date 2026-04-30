#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h chop regime filter
# Donchian breakouts capture momentum moves; volume > 1.5x 20-bar average confirms validity;
# 12h chop index > 61.8 filters for ranging markets (avoid false breakouts in chop).
# Works in bull via upside breakouts, in bear via downside breakouts. Discrete sizing 0.25
# minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_VolumeSpike_12hChop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Calculate 12h Chop Index (choppiness)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = np.abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Chop = 100 * log10(sum(atr,14) / (max(high,14) - min(low,14))) / log10(14)
    highest_14h = df_12h['high'].rolling(window=14, min_periods=14).max().values
    lowest_14h = df_12h['low'].rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_12h / (highest_14h - lowest_14h)) / np.log10(14)
    chop = np.where((highest_14h - lowest_14h) == 0, 100, chop)  # avoid div by zero
    chop = np.nan_to_num(chop, nan=100.0, posinf=100.0, neginf=0.0)
    
    # Align 12h chop to 4h timeframe (wait for completed 12h bar)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_volume_spike = volume_spike[i]
        curr_chop = chop_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and chop < 61.8 (trending market)
            if curr_volume_spike and curr_chop < 61.8:
                # Bullish breakout: price breaks above highest_20
                if curr_close > curr_highest_20:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below lowest_20
                elif curr_close < curr_lowest_20:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below lowest_20 (trailing stop)
            if curr_close < curr_lowest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above highest_20 (trailing stop)
            if curr_close > curr_highest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals