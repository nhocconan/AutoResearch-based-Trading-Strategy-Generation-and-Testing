#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter
# Long when price breaks above 20-period Donchian high + volume > 2.0x 20-period volume avg + price > 1d EMA50
# Short when price breaks below 20-period Donchian low + volume > 2.0x 20-period volume avg + price < 1d EMA50
# Uses 4h price structure (Donchian channels) and 1d EMA for trend alignment
# Designed for low trade frequency (20-50/year) to minimize fee drag while capturing strong trends
# Works in both bull and bear markets by requiring volume confirmation and trend alignment
# Exit when price crosses the opposite Donchian level (reduces whipsaw vs fixed stops)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period volume SMA
    vol_series = pd.Series(volume)
    vol_sma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-period Donchian high
        # 2. Volume confirmation
        # 3. Price above 1d EMA50 (uptrend filter)
        if (close[i] > donchian_high[i]) and vol_confirm and (close[i] > ema_50_1d_aligned[i]):
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-period Donchian low
        # 2. Volume confirmation
        # 3. Price below 1d EMA50 (downtrend filter)
        elif (close[i] < donchian_low[i]) and vol_confirm and (close[i] < ema_50_1d_aligned[i]):
            signals[i] = -0.30
        
        # === EXIT CONDITIONS ===
        # Exit long when price crosses below Donchian low
        # Exit short when price crosses above Donchian high
        elif signals[i-1] > 0 and close[i] < donchian_low[i]:
            signals[i] = 0.0
        elif signals[i-1] < 0 and close[i] > donchian_high[i]:
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian20_Volume_1dEMA50_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0