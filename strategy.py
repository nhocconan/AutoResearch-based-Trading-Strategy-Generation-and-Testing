#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above H3 Camarilla level from prior 1d AND volume > 1.3x 20-period average AND chop < 61.8 (trending)
# - Short when price breaks below L3 Camarilla level from prior 1d AND volume > 1.3x 20-period average AND chop < 61.8 (trending)
# - Exit when price retests H4/L4 levels OR chop > 61.8 (range regime)
# - Uses Camarilla pivot structure for precise entry/exit levels proven effective on ETH
# - Volume confirmation prevents false breakouts, chop filter avoids whipsaws in ranging markets
# - Targets 20-40 trades/year (80-160 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work in both trending and ranging markets when combined with regime filter

name = "4h_1d_camarilla_pivot_breakout_volume_chop_v1"
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
    
    # Pre-compute 1d Camarilla pivot levels (based on prior day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, L3, L4
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4)
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align HTF Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    atr_14 = np.zeros(n)
    tr = np.maximum(prices['high'] - prices['low'], 
                    np.maximum(abs(prices['high'] - prices['close'].shift(1)), 
                              abs(prices['low'] - prices['close'].shift(1))))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    min_low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) == 0, 50, chop)  # avoid division by zero
    chop_trending = chop < 61.8  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_20_avg[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > H3 with volume spike AND trending regime
            if (prices['high'].iloc[i] > camarilla_h3_aligned[i] and 
                vol_spike.iloc[i] and 
                chop_trending[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < L3 with volume spike AND trending regime
            elif (prices['low'].iloc[i] < camarilla_l3_aligned[i] and 
                  vol_spike.iloc[i] and 
                  chop_trending[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests H4/L4 levels (profit target)
            # 2. Chop > 61.8 (range regime - avoid whipsaws)
            if position == 1:  # Long position
                if (prices['low'].iloc[i] <= camarilla_h4_aligned[i] or 
                    chop_trending[i] == False):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['high'].iloc[i] >= camarilla_l4_aligned[i] or 
                    chop_trending[i] == False):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals