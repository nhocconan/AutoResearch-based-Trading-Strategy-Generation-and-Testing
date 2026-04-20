#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + Donchian Breakout with 12h Trend Filter
# The Choppiness Index (CHOP) identifies ranging vs trending markets:
# - CHOP > 61.8 = ranging (mean reversion opportunities)
# - CHOP < 38.2 = trending (trend following opportunities)
# We combine this with Donchian channel breakouts for directional entries in trending markets,
# and mean reversion at Donchian boundaries in ranging markets.
# 12h EMA50 provides higher-timeframe trend filter to avoid counter-trend trades.
# Volume confirmation ensures institutional participation.
# This adaptive approach should work in both bull and bear markets by adapting to market regime.
# Target: 25-40 trades per year to minimize fee drag.

name = "4h_CHOP_Donchian_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === 12h EMA50 for trend direction ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Choppiness Index (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR (14)
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    atr_sum = np.zeros(n)
    for i in range(14, n):
        atr_sum[i] = np.sum(atr[i-13:i+1])
    
    # Max(high) and Min(low) over 14 periods
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(n):
        if i < 14:
            max_high[i] = np.max(high[:i+1])
            min_low[i] = np.min(low[:i+1])
        else:
            max_high[i] = np.max(high[i-13:i+1])
            min_low[i] = np.min(low[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros(n)
    for i in range(14, n):
        if atr_sum[i] > 0 and max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50.0  # Neutral when undefined
    
    # === Donchian Channel (20-period) ===
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
        else:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = close[i]
        ema_val = ema_50_12h_aligned[i]
        chop_val = chop[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is invalid
        if (np.isnan(ema_val) or np.isnan(chop_val) or np.isnan(donchian_high_val) or 
            np.isnan(donchian_low_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trending market (CHOP < 38.2) - Trend following
            if chop_val < 38.2:
                # Long: price breaks above Donchian high with volume, above 12h EMA50
                if close_val > donchian_high_val and vol_ratio_val > 1.5 and close_val > ema_val:
                    signals[i] = 0.30
                    position = 1
                # Short: price breaks below Donchian low with volume, below 12h EMA50
                elif close_val < donchian_low_val and vol_ratio_val > 1.5 and close_val < ema_val:
                    signals[i] = -0.30
                    position = -1
            # Ranging market (CHOP > 61.8) - Mean reversion
            elif chop_val > 61.8:
                # Long: price at Donchian low with volume rejection
                if close_val <= donchian_low_val * 1.001 and vol_ratio_val > 1.5:
                    signals[i] = 0.30
                    position = 1
                # Short: price at Donchian high with volume rejection
                elif close_val >= donchian_high_val * 0.999 and vol_ratio_val > 1.5:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Long exit: reversal signals or volatility expansion
            # Exit when: chop increases (ranging) OR price touches opposite Donchian band
            if chop_val > 50.0 or close_val < donchian_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: reversal signals or volatility expansion
            # Exit when: chop increases (ranging) OR price touches opposite Donchian band
            if chop_val > 50.0 or close_val > donchian_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals