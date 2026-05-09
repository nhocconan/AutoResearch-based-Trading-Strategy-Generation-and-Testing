#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.5x average).
# Donchian breakouts capture momentum; EMA50 filters for trend direction; volume confirms strength.
# Designed to work in both bull and bear markets by taking breakouts in direction of higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    # For breakout, we need the highest high and lowest low of the past 20 periods excluding current
    # We'll use rolling window via convolution-like approach but simpler: use pandas rolling then align
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values  # shift(1) to avoid lookahead
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and 20 for Donchian (with shift)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_50_1d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for confirmation
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: price > Donchian upper AND price > 1d EMA50 (uptrend) AND volume > 1.5x average
            if close[i] > upper and close[i] > ema_1d and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Donchian lower AND price < 1d EMA50 (downtrend) AND volume > 1.5x average
            elif close[i] < lower and close[i] < ema_1d and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian lower OR trend reverses (price < 1d EMA50)
            if close[i] < lower or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian upper OR trend reverses (price > 1d EMA50)
            if close[i] > upper or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals