#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 20-bar Donchian breakout with 1w EMA50 trend filter and volume confirmation
# Uses weekly EMA50 for trend alignment to avoid counter-trend trades
# Breakout confirmed by volume > 2x 24-bar average (48h volume spike)
# ATR-based trailing stop via signal=0 when price retraces 30% of ATR from extreme
# Discrete sizing 0.25 to balance profit potential and fee drag; target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear: breakouts capture momentum, trend filter avoids counter-trend traps, volume filter ensures participation
# Weekly trend filter provides stronger bias than daily, reducing whipsaw in sideways markets

name = "6h_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>2x 24-bar average = 48h volume spike)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma_24)
    
    # Calculate Donchian channels (20-bar)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 6h timeframe (primary)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute session filter (00-24 UTC - trade all hours for 6h)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = np.ones(n, dtype=bool)  # trade all hours
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long breakout: price > 20-bar high AND uptrend (price > weekly EMA50) AND volume spike
            if close[i] > highest_high[i] and close[i] > ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short breakdown: price < 20-bar low AND downtrend (price < weekly EMA50) AND volume spike
            elif close[i] < lowest_low[i] and close[i] < ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 30% of ATR from extreme
            if close[i] <= long_extreme - 0.30 * atr[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 30% of ATR from extreme
            if close[i] >= short_extreme + 0.30 * atr[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals