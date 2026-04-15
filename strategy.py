#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period volume avg + price > 1d EMA50
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period volume avg + price < 1d EMA50
# Uses Donchian channels for structure and 1d EMA for multi-timeframe trend alignment
# Designed for low trade frequency (20-40/year) to minimize fee drag and maximize test generalization
# Volume confirmation reduces false breakouts
# Works in both bull and bear markets by requiring volume and trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    # Note: We use 4h data but since primary timeframe is 4h, we work directly on prices
    # For 4h timeframe, the prices DataFrame already contains 4h data
    high_4h = high
    low_4h = low
    close_4h = close
    
    # Calculate Donchian Channel (20-period high/low)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period volume SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-period Donchian high
        # 2. Volume confirmation
        # 3. Price above 1d EMA50 (long-term uptrend)
        if (close[i] > donchian_high[i]) and vol_confirm and \
           (close[i] > ema_50_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-period Donchian low
        # 2. Volume confirmation
        # 3. Price below 1d EMA50 (long-term downtrend)
        elif (close[i] < donchian_low[i]) and vol_confirm and \
             (close[i] < ema_50_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_1dEMA50_Filter_v2"
timeframe = "4h"
leverage = 1.0