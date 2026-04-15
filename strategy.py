#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d chop regime filter
# Long when price breaks above Donchian high + volume > 1.8x 20-bar avg + choppy market (CHOP > 61.8)
# Short when price breaks below Donchian low + volume > 1.8x 20-bar avg + choppy market (CHOP > 61.8)
# Uses strict volume filter (1.8x) to reduce trades to target range (20-40/year)
# Chop regime ensures breakouts occur in ranging markets where mean reversion fails, increasing edge
# Designed for low trade frequency to minimize fee drag while capturing asymmetric breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channel (20) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # === 1d Indicators: ATR for Choppy Market Calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = high_1d[0] - low_1d[0]
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Choppy Market Calculation: CHOP = 100 * log10(sum(ATR14) / (max(high)-min(low))) / log10(14)
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero and invalid values
    denominator = max_high_14 - min_low_14
    chop_raw = np.where(denominator > 0, sum_atr_14 / denominator, 1.0)
    choppy_market = 100 * np.log10(np.maximum(chop_raw, 1e-10)) / np.log10(14)
    choppy_market_aligned = align_htf_to_ltf(prices, df_1d, choppy_market)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.8x 20-period volume SMA (stricter to reduce trades)
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(choppy_market_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h upper Donchian
        # 2. Volume confirmation (strict 1.8x)
        # 3. Choppy market regime (CHOP > 61.8 = ranging/mean reverting)
        if (close[i] > donchian_high_aligned[i]) and vol_confirm and (choppy_market_aligned[i] > 61.8):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h lower Donchian
        # 2. Volume confirmation (strict 1.8x)
        # 3. Choppy market regime (CHOP > 61.8 = ranging/mean reverting)
        elif (close[i] < donchian_low_aligned[i]) and vol_confirm and (choppy_market_aligned[i] > 61.8):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_VolumeStrict_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0