#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation.
# Long when price breaks above 1d Donchian upper channel AND close > 1w EMA50 (uptrend) AND volume > 1.5x average.
# Short when price breaks below 1d Donchian lower channel AND close < 1w EMA50 (downtrend) AND volume > 1.5x average.
# Exit when price crosses 1d Donchian midpoint (mean reversion) or volume drops below average.
# Uses discrete position size 0.25. Donchian channels provide clear breakout levels with built-in stop via midpoint.
# 1w EMA50 ensures trading only with higher timeframe trend to avoid whipsaws in choppy markets.
# Volume confirmation filters out weak breakouts. 4h timeframe targets 75-200 total trades over 4 years (19-50/year).
# Works in bull markets (catch breakouts in uptrends) and bear markets (catch breakdowns in downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian(20)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Donchian(20) channels ===
    # Upper channel = highest high over past 20 days
    # Lower channel = lowest low over past 20 days
    # Middle channel = (upper + lower) / 2
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (4h)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_1d, mid_20)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol = volume[i]
        upper = high_20_aligned[i]
        lower = low_20_aligned[i]
        mid = mid_20_aligned[i]
        ema50 = ema50_aligned[i]
        vol_avg = vol_ma[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < midpoint (mean reversion) OR volume < average (weak momentum)
            if (price < mid) or (vol < vol_avg):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > midpoint (mean reversion) OR volume < average (weak momentum)
            if (price > mid) or (vol < vol_avg):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper channel AND price > EMA50 (uptrend) AND volume > 1.5x average
            if (price > upper) and (price > ema50) and (vol > 1.5 * vol_avg):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower channel AND price < EMA50 (downtrend) AND volume > 1.5x average
            elif (price < lower) and (price < ema50) and (vol > 1.5 * vol_avg):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dDonchian20_1wEMA50_VolumeConfirmation_V1"
timeframe = "4h"
leverage = 1.0