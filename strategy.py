#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with volume confirmation and 12h EMA50 trend filter
# Long when price breaks above 20-bar Donchian high + volume > 2.0x 20-period avg + 12h EMA50 > EMA50 prev bar (uptrend)
# Short when price breaks below 20-bar Donchian low + volume > 2.0x 20-period avg + 12h EMA50 < EMA50 prev bar (downtrend)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Donchian channels provide clear breakout levels. Volume confirms breakout strength. 12h EMA50 ensures we trade with higher timeframe trend.
# Works in bull markets (trend continuation) and bear markets (strong downtrends) by requiring alignment with 12h EMA50 slope.

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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicator: EMA50 (trend direction) ===
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate EMA50 slope (current > previous = uptrend)
    ema50_slope = np.zeros_like(ema50_12h_aligned)
    ema50_slope[1:] = ema50_12h_aligned[1:] > ema50_12h_aligned[:-1]
    
    # === 6h Indicators: Donchian(20) channels ===
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    lookback = 20
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(100, lookback-1, 50)  # ensure Donchian and EMA50 are ready
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_slope[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-bar Donchian high
        # 2. Uptrend (12h EMA50 sloping up)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           ema50_slope[i] and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-bar Donchian low
        # 2. Downtrend (12h EMA50 sloping down)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             (not ema50_slope[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_Volume_12hEMA50Trend_v1"
timeframe = "6h"
leverage = 1.0