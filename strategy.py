#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w EMA100 trend filter
# Long when price breaks above 20-period Donchian high + volume > 2.0x 20-period avg + price > 1w EMA100
# Short when price breaks below 20-period Donchian low + volume > 2.0x 20-period avg + price < 1w EMA100
# Uses daily price structure (Donchian channels) and weekly EMA for trend alignment
# Designed for very low trade frequency (10-25/year) to minimize fee drag while capturing strong trends
# Volume confirmation and weekly trend filter reduce false breakouts in choppy markets
# Works in both bull and bear markets by requiring volume confirmation and weekly trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # === 1w Indicators: EMA100 ===
    close_1w = df_1w['close'].values
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Donchian(20) channels
        donchian_high = np.max(high[i-19:i+1]) if i >= 19 else np.nan
        donchian_low = np.min(low[i-19:i+1]) if i >= 19 else np.nan
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high) or np.isnan(donchian_low) or
            np.isnan(ema_100_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-period Donchian high
        # 2. Volume confirmation
        # 3. Price above 1w EMA100 (uptrend filter)
        if (close[i] > donchian_high) and vol_confirm and (close[i] > ema_100_1w_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-period Donchian low
        # 2. Volume confirmation
        # 3. Price below 1w EMA100 (downtrend filter)
        elif (close[i] < donchian_low) and vol_confirm and (close[i] < ema_100_1w_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_Volume_1wEMA100_Filter_v1"
timeframe = "1d"
leverage = 1.0