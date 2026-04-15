#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian upper band (20-period high) + weekly pivot bullish + volume > 1.5x 20-period avg
# Short when price breaks below Donchian lower band (20-period low) + weekly pivot bearish + volume > 1.5x 20-period avg
# Weekly pivot direction derived from prior week's close: bullish if prior weekly close > prior weekly open
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Weekly pivot provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~15-35 trades/year on 6h timeframe to avoid overtrading.
# Donchian channels provide clear structure-based breakout levels that work in ranging and trending markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1w Indicator: Weekly Pivot Direction ===
    # Bullish if prior weekly close > prior weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish week
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # === 6h Donchian Channel (20-period) ===
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    # Using rolling window with min_periods to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(10, 20) + 5  # Weekly data + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Weekly pivot direction: 1.0 = bullish, 0.0 = bearish
        weekly_dir = weekly_bullish_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper band (close > upper)
        # 2. Weekly pivot bullish (prior weekly close > prior weekly open)
        # 3. Volume confirmation
        if (close[i] > donchian_upper[i]) and \
           (weekly_dir > 0.5) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower band (close < lower)
        # 2. Weekly pivot bearish (prior weekly close <= prior weekly open)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower[i]) and \
             (weekly_dir <= 0.5) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_1wPivot_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0