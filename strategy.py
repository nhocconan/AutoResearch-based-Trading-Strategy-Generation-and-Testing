#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_12h
Hypothesis: Donchian(20) breakout with volume confirmation and 12h EMA trend filter.
Long when price breaks above upper band with volume spike in uptrend.
Short when price breaks below lower band with volume spike in downtrend.
Designed for low trade frequency (<25/year) to avoid fee decay while capturing strong trends.
Works in both bull and bear markets by following the 12h trend direction.
"""

name = "4h_Donchian_Breakout_Volume_Trend_12h"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h EMA21 for trend filter ---
    close_12h = df_12h['close']
    ema_21_12h = close_12h.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # --- Donchian Channel (20-period) ---
    lookback = 20
    # Calculate highest high and lowest low over lookback period
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # --- Volume Spike Detection (1.5x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ema.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on price vs 12h EMA
        price_above_ema = close[i] > ema_21_12h_aligned[i]
        price_below_ema = close[i] < ema_21_12h_aligned[i]
        
        # Breakout conditions
        long_breakout = (high[i] > highest_high[i]) and vol_spike[i]
        short_breakout = (low[i] < lowest_low[i]) and vol_spike[i]
        
        if position == 0:
            if price_above_ema:
                # Uptrend: only take long breakouts
                if long_breakout:
                    signals[i] = 0.25
                    position = 1
            elif price_below_ema:
                # Downtrend: only take short breakouts
                if short_breakout:
                    signals[i] = -0.25
                    position = -1
            # If price is near EMA, stay flat to avoid whipsaws
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price closes below Donchian lower band or trend changes
                if close[i] < lowest_low[i] or not price_above_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price closes above Donchian upper band or trend changes
                if close[i] > highest_high[i] or not price_below_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals