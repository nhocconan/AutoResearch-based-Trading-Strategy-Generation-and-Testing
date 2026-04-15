#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band (20-period high) + price > 12h EMA50 + volume > 1.5x 20-period avg
# Short when price breaks below 4h Donchian lower band (20-period low) + price < 12h EMA50 + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
# Donchian channels provide clear trend structure; 12h EMA ensures alignment with higher timeframe trend.
# Volume confirmation filters out weak breakouts. Works in bull markets (continuation) and bear markets (strong downtrends).

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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicator: Donchian Channels (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian upper/lower bands: highest high/lowest low over 20 periods
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h Indicator: EMA(50) for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 4h Indicator: Volume SMA (20-period) for confirmation ===
    vol_sma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20_4h[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_sma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper band
        # 2. Price above 12h EMA50 (uptrend alignment)
        # 3. Volume confirmation
        if (close[i] > donchian_upper_aligned[i]) and \
           (close[i] > ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower band
        # 2. Price below 12h EMA50 (downtrend alignment)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower_aligned[i]) and \
             (close[i] < ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_EMA50_12h_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0