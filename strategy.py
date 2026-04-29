#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and chop regime filter
# Donchian(20) captures price channel breakouts; volume > 1.5x 20-period MA confirms strength
# Choppiness Index (CHOP) > 61.8 = ranging (mean revert at bands), CHOP < 38.2 = trending (breakout follow)
# Works in both bull/bear by adapting to regime: breakout in trending, mean revert in ranging
# Target: 20-50 trades/year (80-200 total over 4 years)

name = "4h_Donchian20_Breakout_VolumeChop_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (HH(14) - LL(14))))
    # Simplified: CHOP = 100 * LOG10( SUM(TR(14)) / (LOG10(14) * (MAX(HIGH,14) - MIN(LOW,14))) )
    tr1 = pd.Series(df_1d['high']).rolling(14, min_periods=14).max() - pd.Series(df_1d['low']).rolling(14, min_periods=14).min()
    tr2 = abs(pd.Series(df_1d['high']).rolling(14, min_periods=14).max() - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']).rolling(14, min_periods=14).min() - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high']).rolling(14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (np.log10(14) * (highest_high_14 - lowest_low_14)))
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop_raw)  # neutral when no range
    
    # Align chop to 4h timeframe (completed 1d bar only)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_chop = chop_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        # Regime determination
        # CHOP > 61.8 = ranging (mean revert)
        # CHOP < 38.2 = trending (follow breakout)
        # 38.2 <= CHOP <= 61.8 = transition (no clear signal)
        
        if position == 0:  # Flat - look for new entries
            if curr_chop < 38.2:  # Trending regime - follow breakout
                # Long on upside breakout with volume
                if curr_high > curr_highest_high and curr_volume_confirm:
                    signals[i] = 0.30
                    position = 1
                # Short on downside breakout with volume
                elif curr_low < curr_lowest_low and curr_volume_confirm:
                    signals[i] = -0.30
                    position = -1
                    
            elif curr_chop > 61.8:  # Ranging regime - mean revert at bands
                # Long near lower band with volume
                if curr_low <= curr_lowest_low * 1.001 and curr_volume_confirm:  # near lower band
                    signals[i] = 0.30
                    position = 1
                # Short near upper band with volume
                elif curr_high >= curr_highest_high * 0.999 and curr_volume_confirm:  # near upper band
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit on downside break of lower band OR chop > 61.8 (range) with price near middle
            if curr_low < curr_lowest_low or (curr_chop > 61.8 and curr_close > (curr_highest_high + curr_lowest_low) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position - exit conditions
            # Exit on upside break of upper band OR chop > 61.8 (range) with price near middle
            if curr_high > curr_highest_high or (curr_chop > 61.8 and curr_close < (curr_highest_high + curr_lowest_low) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals