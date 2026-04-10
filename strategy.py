#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 (1d) AND 1d volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 (1d) AND 1d volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
# - Exit when price returns to Camarilla pivot point (1d)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla pivots from 1d provide intraday support/resistance levels
# - Volume confirmation ensures breakout validity
# - Chop filter avoids false breakouts in ranging markets
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Resistance levels
    h3 = pivot + (range_hl * 1.1 / 4)
    h4 = pivot + (range_hl * 1.1 / 2)
    # Support levels
    l3 = pivot - (range_hl * 1.1 / 4)
    l4 = pivot - (range_hl * 1.1 / 2)
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 12h chop regime filter: CHOP(14) < 61.8 = trending
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range calculation
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = log10(sum(ATR14)/ (max(high)-min(low)))*100/(log10(14))
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    range_14 = highest_high - lowest_low
    chop = np.where(range_14 > 0, 
                    np.log10(atr_sum) / np.log10(range_14) * 100 / np.log10(14), 
                    50)
    chop = np.nan_to_num(chop, nan=50.0)
    chop_filter = chop < 61.8  # Trending regime
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_spike_aligned[i]) or
            np.isnan(chop_filter[i])):
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
            if (close[i] > h3_aligned[i] and 
                vol_spike_aligned[i] and 
                chop_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 AND volume spike AND trending regime
            elif (close[i] < l3_aligned[i] and 
                  vol_spike_aligned[i] and 
                  chop_filter[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point
            # Exit when price returns to pivot point (mean reversion to equilibrium)
            exit_long = position == 1 and close[i] <= pivot_aligned[i]
            exit_short = position == -1 and close[i] >= pivot_aligned[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals