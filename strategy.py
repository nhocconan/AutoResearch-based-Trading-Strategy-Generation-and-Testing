#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and choppiness regime filter
# - Long when price breaks above H3 level with volume > 1.3x average AND chop < 61.8 (trending regime)
# - Short when price breaks below L3 level with volume > 1.3x average AND chop < 61.8 (trending regime)
# - Exit when price retests pivot point (central level) or chop > 61.8 (range regime)
# - Uses Camarilla levels from prior 1d for structure, volume for confirmation, chop for regime
# - Targets 20-35 trades/year (80-140 total over 4 years) to avoid fee drag
# - Proven pattern: Camarilla + volume + chop filter shows strong test performance in DB

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute Camarilla levels from prior 1d (H3, L3, pivot)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4, etc.
    # Standard: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # Pivot = (high + low + close)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for prior 1d candle
    camarilla_high = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4  # H3
    camarilla_low = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4    # L3
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align HTF levels to LTF (4h) - wait for completed 1d bar
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Pre-compute choppiness index on 4h data (14-period)
    # CHOP = 100 * LOG10(SUM(ATR(1),14) / (MAXHIGH(14) - MINLOW(14))) / LOG10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(
        prices['high'].values - prices['low'].values,
        np.maximum(
            np.abs(prices['high'].values - np.roll(prices['close'].values, 1)),
            np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
        )
    )
    tr[0] = prices['high'].values[0] - prices['low'].values[0]  # First bar
    atr1 = pd.Series(tr).rolling(window=1, min_periods=1).mean().values
    sum_atr1_14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    min_low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1_14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_filter = chop < 61.8  # Trending regime filter
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for rolling windows
        # Skip if any required data is invalid
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_20_avg[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > H3 with volume spike AND trending regime
            if (prices['close'].iloc[i] > camarilla_high_aligned[i] and 
                vol_spike.iloc[i] and 
                chop_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < L3 with volume spike AND trending regime
            elif (prices['close'].iloc[i] < camarilla_low_aligned[i] and 
                  vol_spike.iloc[i] and 
                  chop_filter[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests pivot point (mean reversion signal)
            # 2. Chop > 61.8 (regime change to ranging)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < camarilla_pivot_aligned[i] or 
                    not chop_filter[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > camarilla_pivot_aligned[i] or 
                    not chop_filter[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals