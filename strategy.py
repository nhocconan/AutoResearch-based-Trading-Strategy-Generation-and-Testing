#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Long when price breaks above 12h Donchian upper(20) + volume > 1.5x 20-period avg + 1d ATR(14) > 0.5 * price (sufficient volatility)
# Short when price breaks below 12h Donchian lower(20) + same filters
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-25/year).
# Donchian channels provide clear breakout levels. ATR filter ensures we only trade when volatility is sufficient to avoid whipsaws.
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation) by requiring volatility filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 12h Indicator: Donchian Channel (20-period) ===
    lookback = 20
    # Calculate rolling max/min for Donchian bands
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1d Indicator: ATR (14-period) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's ATR smoothing
    period = 14
    atr_1d = np.zeros_like(tr)
    atr_1d[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * (period-1) + tr[i]) / period
    
    # Align 1d ATR to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(lookback, 30)  # Ensure Donchian and ATR are ready
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Volatility filter: 1d ATR > 0.5% of price (ensures sufficient volatility)
        vol_filter = atr_1d_aligned[i] > (close[i] * 0.005)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 12h Donchian upper(20)
        # 2. Volume confirmation
        # 3. Sufficient volatility
        if (close[i] > highest[i]) and \
           vol_confirm and vol_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 12h Donchian lower(20)
        # 2. Volume confirmation
        # 3. Sufficient volatility
        elif (close[i] < lowest[i]) and \
             vol_confirm and vol_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_Volume_ATR_VolFilter_v1"
timeframe = "12h"
leverage = 1.0