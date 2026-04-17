#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with weekly EMA trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND weekly EMA50 is rising AND volume > 1.5x 20-day average.
Short when price breaks below Donchian lower band AND weekly EMA50 is falling AND volume > 1.5x 20-day average.
Exit when price retraces 50% of ATR from the extreme favorable price since entry.
Uses weekly timeframe for trend filter to avoid counter-trend trades in bear markets.
Designed for low trade frequency (7-25/year) to minimize fee drag while capturing strong breakouts.
Works in both bull (breakouts with trend) and bear (avoids false breakouts via weekly trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on daily timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period) on daily
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) on daily for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar: use high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    extreme_price = 0.0  # Tracks best price since entry for trailing stop
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_ma = volume_ma[i]
        vol = volume[i]
        atr_val = atr[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        ema_1w = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band AND weekly EMA50 rising AND volume > 1.5x avg
            if high_price > upper and ema_1w > close_1w[-1] if len(close_1w) > 0 else False and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                extreme_price = price
            # Short: price breaks below lower band AND weekly EMA50 falling AND volume > 1.5x avg
            elif low_price < lower and ema_1w < close_1w[-1] if len(close_1w) > 0 else False and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                extreme_price = price
        
        elif position == 1:
            # Update extreme price (highest since entry)
            if price > extreme_price:
                extreme_price = price
            # Exit long: price retraces 50% of ATR from extreme price
            if price < extreme_price - 0.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update extreme price (lowest since entry)
            if price < extreme_price:
                extreme_price = price
            # Exit short: price retraces 50% of ATR from extreme price
            if price > extreme_price + 0.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA50_Volume_ATRTrail"
timeframe = "1d"
leverage = 1.0