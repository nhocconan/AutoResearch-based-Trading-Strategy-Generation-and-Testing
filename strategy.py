#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-bar avg AND choppiness < 61.8 (trending)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-bar avg AND choppiness < 61.8
# - Exit when price returns to Camarilla Pivot level (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla pivots identify key intraday support/resistance levels from prior day
# - Volume confirmation avoids low-liquidity false breakouts
# - Choppiness filter ensures we only trade in trending markets (avoids chop)
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Works in both bull and bear markets: breakouts work in trends, pivot mean reversion works in ranges

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    # H4 = close + 1.5*(high-low)*1.1/2, H3 = close + 1.25*(high-low)*1.1/2
    # L3 = close - 1.25*(high-low)*1.1/2, L4 = close - 1.5*(high-low)*1.1/2
    # Pivot = (high + low + close)/3
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.25 * range_1d * 1.1 / 2
    camarilla_l3 = close_1d - 1.25 * range_1d * 1.1 / 2
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align 1d Camarilla levels to 4h timeframe (previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute choppiness regime filter on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: 100 * log10(sum(ATR)/ (n * (max(high)-min(low)))) / log10(n)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denominator = 14 * (max_high - min_low)
    chop_ratio = np.where(chop_denominator > 0, sum_atr / chop_denominator, 1.0)
    chop_ratio = np.where(chop_ratio > 0, chop_ratio, 1e-10)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Choppiness < 61.8 = trending (regime filter)
    trending_regime = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(trending_regime[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND volume spike AND trending regime
            if (close[i] > camarilla_h3_aligned[i] and 
                vol_spike_1d_aligned[i] and 
                trending_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND volume spike AND trending regime
            elif (close[i] < camarilla_l3_aligned[i] and 
                  vol_spike_1d_aligned[i] and 
                  trending_regime[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Camarilla Pivot (mean reversion)
            # Exit when price returns to Pivot level
            exit_signal = np.abs(close[i] - camarilla_pivot_aligned[i]) < 0.1 * camarilla_pivot_aligned[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals