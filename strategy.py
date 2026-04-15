#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(40) breakout with volume confirmation and 1w EMA100 trend filter
# Long when price breaks above 40-period Donchian high + volume > 2.0x 20-period avg + price > 1w EMA100
# Short when price breaks below 40-period Donchian low + volume > 2.0x 20-period avg + price < 1w EMA100
# Uses longer Donchian period (40) for fewer, stronger breakouts and higher volume threshold (2.0x) to reduce false signals
# 1w EMA100 provides strong trend filter to avoid counter-trend trades in choppy markets
# Designed for very low trade frequency (<10/year) to minimize fee drag while capturing major trends
# Works in both bull and bear markets by requiring strong volume confirmation and trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (40-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high/low (40-period)
    donchian_high = pd.Series(high_1d).rolling(window=40, min_periods=40).max().values
    donchian_low = pd.Series(low_1d).rolling(window=40, min_periods=40).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1d Indicators: Volume SMA (20-period) ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1w Indicators: EMA100 for Trend Filter ===
    close_1w = df_1w['close'].values
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 2.0x 20-period volume SMA (strict threshold)
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_100_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Donchian high (40-period)
        # 2. Volume confirmation (strict 2.0x threshold)
        # 3. Price above 1w EMA100 (strong uptrend filter)
        if (close[i] > donchian_high_aligned[i]) and vol_confirm and (close[i] > ema_100_1w_aligned[i]):
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Donchian low (40-period)
        # 2. Volume confirmation (strict 2.0x threshold)
        # 3. Price below 1w EMA100 (strong downtrend filter)
        elif (close[i] < donchian_low_aligned[i]) and vol_confirm and (close[i] < ema_100_1w_aligned[i]):
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian40_Volume2x_1wEMA100_Filter_v1"
timeframe = "1d"
leverage = 1.0