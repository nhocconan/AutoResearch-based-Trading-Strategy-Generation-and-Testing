#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index (14) + 12h Donchian Breakout (20) + Volume Spike
# In choppy markets (CHOP > 61.8), mean-reversion at Donchian bands works well.
# Volume spike confirms breakout validity. Uses 1w trend filter to avoid counter-trend trades.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (~12-37/year).
# Works in bull/bear via trend filter and regime adaptation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA trend filter (50-period)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if atr[i] > 0 and max_h[i] > min_l[i]:
            chop[i] = 100 * np.log14(np.sum(atr[i-13:i+1]) / (max_h[i] - min_l[i])) / np.log14(14)
        else:
            chop[i] = 50.0  # neutral when undefined
    
    # Donchian Channels (20-period)
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need chop (14), DC (20), EMA (50), vol MA (20)
    start_idx = max(14, 20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop[i]) or np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 1w EMA
        bullish_trend = price > ema_50_1w_aligned[i]
        bearish_trend = price < ema_50_1w_aligned[i]
        
        # Regime filter: choppy market (CHOP > 61.8) for mean reversion
        chop_filter = chop[i] > 61.8
        
        if position == 0:
            # Long: price at lower Donchian band in choppy market with volume
            if chop_filter and vol_filter and price <= dc_low[i]:
                signals[i] = size
                position = 1
            # Short: price at upper Donchian band in choppy market with volume
            elif chop_filter and vol_filter and price >= dc_high[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches middle or upper band, or trend turns bearish
            dc_mid = (dc_high[i] + dc_low[i]) / 2
            if price >= dc_mid or not bullish_trend or not chop_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches middle or lower band, or trend turns bullish
            dc_mid = (dc_high[i] + dc_low[i]) / 2
            if price <= dc_mid or not bearish_trend or not chop_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Choppiness_Donchian_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0