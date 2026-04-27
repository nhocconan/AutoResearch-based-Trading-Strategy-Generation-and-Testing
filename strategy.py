#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian breakout and volume confirmation
# In high volatility/trending markets (CHOP < 38.2), trade Donchian breakouts with volume
# In ranging markets (CHOP > 61.8), avoid trades to prevent whipsaw
# Works in bull/bear by adapting to market regime via Choppiness Index
# Target: 75-200 total trades over 4 years (~19-50/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Choppiness Index calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range and ATR for Choppiness Index
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14 = np.full(len(df_4h), np.nan)
    for i in range(14, len(tr)):
        atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # Calculate Choppiness Index
    chop = np.full(len(df_4h), np.nan)
    for i in range(14, len(df_4h)):
        atr_sum = np.nansum(atr_14[i-13:i+1])
        hh = np.nanmax(high_4h[i-13:i+1])
        ll = np.nanmin(low_4h[i-13:i+1])
        if atr_sum > 0 and hh > ll:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe (wait for 4h close)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    # Donchian channels (20-period) on 4h
    donchian_high = np.full(len(df_4h), np.nan)
    donchian_low = np.full(len(df_4h), np.nan)
    for i in range(20, len(df_4h)):
        donchian_high[i] = np.nanmax(high_4h[i-20:i])
        donchian_low[i] = np.nanmin(low_4h[i-20:i])
    
    # Align Donchian channels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 4h data for Choppiness (14), ATR (14), Donchian (20), volume MA (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        # Volume filter: significant volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: break above Donchian high with volume and trending regime
            if price > donchian_high_aligned[i] and vol_filter and trending_regime:
                signals[i] = size
                position = 1
            # Short: break below Donchian low with volume and trending regime
            elif price < donchian_low_aligned[i] and vol_filter and trending_regime:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low or regime changes to ranging
            if price <= donchian_low_aligned[i] or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high or regime changes to ranging
            if price >= donchian_high_aligned[i] or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Choppiness_Donchian_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0