#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout with volume confirmation
# In high chop (range), mean-revert at Donchian bands; in low chop (trend), follow breakouts
# Uses 1d Donchian(20) for structure, 4h Choppiness Index(14) for regime, and 1d volume for confirmation
# Designed to work in both bull and bear markets by adapting to market regime
# Target: 20-40 trades per symbol over 4 years (5-10/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Choppiness Index (14-period)
    # CHOP = 100 * log10(SUM(TR,14) / (HHV(H,14) - LLV(L,14))) / log10(14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Donchian and Chop calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # In high chop (range): mean revert at bands
            if chop_val > 61.8:  # Choppy market
                if price <= donchian_low_aligned[i] and vol_1d_current > 1.3 * vol_ma_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                elif price >= donchian_high_aligned[i] and vol_1d_current > 1.3 * vol_ma_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:  # Trending market
                if price > donchian_high_aligned[i] and vol_1d_current > 1.5 * vol_ma_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                elif price < donchian_low_aligned[i] and vol_1d_current > 1.5 * vol_ma_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses mid or volatility drops
            if price >= donchian_mid_aligned[i] or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses mid or volatility drops
            if price <= donchian_mid_aligned[i] or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Chop_Donchian_MeanRev_Trend"
timeframe = "4h"
leverage = 1.0