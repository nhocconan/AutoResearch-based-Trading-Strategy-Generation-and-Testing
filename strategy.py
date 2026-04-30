#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR trailing stop
# Donchian channels identify key breakout levels where institutional order flow accumulates.
# Breakouts above/below 20-period high/low with volume spike indicate strong momentum.
# ATR trailing stop allows profits to run while limiting drawdowns in both bull and bear markets.
# Designed for moderate trade frequency (~30-50/year) to balance signal quality and fee drag.
# Uses 4h timeframe with 1d HTF for volume regime filter (avoid low-volume false breakouts).

name = "4h_Donchian20_Breakout_VolumeRegime_ATRTrail_v1"
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
    
    # Load 1d data ONCE before loop for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d average volume for regime filter (avoid low-volume environment)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(14) for Donchian channels and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    # Upper channel: highest high over past 20 periods
    # Lower channel: lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    start_idx = 20  # warmup for Donchian channels
    
    for i in range(start_idx, n):
        # Volume regime filter: avoid trading in low-volume environments
        # Only trade when current 12h volume > 20-period 1d average volume
        volume_regime_ok = volume[i] > vol_ma_20_1d_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume regime confirmation
            if volume_regime_ok:
                # Bullish entry: price breaks above Donchian upper channel
                if curr_close > curr_upper:
                    signals[i] = 0.30
                    position = 1
                    highest_since_entry = curr_close
                # Bearish entry: price breaks below Donchian lower channel
                elif curr_close < curr_lower:
                    signals[i] = -0.30
                    position = -1
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry for trailing stop
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # ATR trailing stop: 2.5 * ATR below highest price since entry
            if curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Optional: exit if price re-enters Donchian channel (mean reversion)
            elif curr_close < curr_upper and curr_close > curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Update lowest price since entry for trailing stop
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # ATR trailing stop: 2.5 * ATR above lowest price since entry
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Optional: exit if price re-enters Donchian channel (mean reversion)
            elif curr_close > curr_lower and curr_close < curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals