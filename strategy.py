#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 level with volume > 1.3x average AND chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 level with volume > 1.3x average AND chop < 61.8 (trending)
# - Exit when price returns to Camarilla Pivot level or chop > 61.8 (range)
# - Uses 1d Camarilla levels for structure, 12h for entry timing, volume confirmation to avoid false breakouts
# - Chop filter ensures we only trade in trending markets, avoiding whipsaws in ranges
# - Targets 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in both trending and ranging markets when combined with volume and regime filters

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, H2, H1, Pivot, L1, L2, L3, L4
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    # Pivot = (High + Low + Close) / 3
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2.0
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align 1d Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Pre-compute 12h volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute chop regime filter (14-period) on 12h data
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    tr1 = np.maximum(prices['high'], prices['close'].shift(1)) - np.minimum(prices['low'], prices['close'].shift(1))
    tr2 = np.maximum(prices['low'], prices['close'].shift(1)) - np.minimum(prices['high'], prices['close'].shift(1))
    tr = np.maximum(tr1, tr2)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / (high_14 - low_14)) / np.log10(14)
    chop_trending = chop < 61.8  # Trending regime
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_20_avg[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Camarilla H3 with volume spike AND trending regime
            if (prices['high'].iloc[i] > camarilla_h3_aligned[i] and 
                vol_spike.iloc[i] and 
                chop_trending[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < Camarilla L3 with volume spike AND trending regime
            elif (prices['low'].iloc[i] < camarilla_l3_aligned[i] and 
                  vol_spike.iloc[i] and 
                  chop_trending[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to Camarilla Pivot level (mean reversion)
            # 2. Chop > 61.8 (regime change to range)
            if position == 1:  # Long position
                if (prices['low'].iloc[i] <= camarilla_pivot_aligned[i] or 
                    chop_trending[i] == False):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['high'].iloc[i] >= camarilla_pivot_aligned[i] or 
                    chop_trending[i] == False):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals