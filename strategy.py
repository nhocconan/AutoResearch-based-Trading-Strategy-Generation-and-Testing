#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + chop regime filter
# - Uses 1d Camarilla pivot levels (L3, L4, H3, H4) for mean reversion entries
# - Long when price touches L3/L4 with volume spike and chop > 61.8 (range)
# - Short when price touches H3/H4 with volume spike and chop > 61.8 (range)
# - Exits when price reaches opposite Camarilla level (H3/H4 for long, L3/L4 for short)
# - Position size: 0.25 (25% of capital) to minimize drawdown in bear markets
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize fee drag
# - Works in ranging markets (mean reversion at pivots) and avoids trending markets via chop filter

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ATR and chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(14) for stoploss
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Camarilla pivot levels (based on previous day)
    # Camarilla levels: H4 = close + 1.1*(high-low)*1.5, H3 = close + 1.1*(high-low)*1.25
    #                   L3 = close - 1.1*(high-low)*1.25, L4 = close - 1.1*(high-low)*1.5
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * range_1d * 1.5
    camarilla_h3 = close_1d + 1.1 * range_1d * 1.25
    camarilla_l3 = close_1d - 1.1 * range_1d * 1.25
    camarilla_l4 = close_1d - 1.1 * range_1d * 1.5
    
    # 1d Volume > 1.8x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.8 * avg_volume_20)
    
    # 1d Choppiness Index(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.where((highest_14 - lowest_14) > 0, highest_14 - lowest_14, 1e-10)
    chop = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    chop_range = chop > 61.8  # range-bound market
    
    # Align all 1d indicators to 12h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range.astype(float))
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_range_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Camarilla level (H3) or ATR stoploss
            if high[i] >= camarilla_h3_aligned[i]:  # Touch H3 level
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.5 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Camarilla level (L3) or ATR stoploss
            if low[i] <= camarilla_l3_aligned[i]:  # Touch L3 level
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.5 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touch with volume confirmation and chop regime
            # Long when price touches L3/L4 with volume spike in range market
            if ((low[i] <= camarilla_l3_aligned[i] or low[i] <= camarilla_l4_aligned[i]) and
                volume_spike_aligned[i] and
                chop_range_aligned[i]):
                position = 1
                entry_price = low[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = 0.25
            # Short when price touches H3/H4 with volume spike in range market
            elif ((high[i] >= camarilla_h3_aligned[i] or high[i] >= camarilla_h4_aligned[i]) and
                  volume_spike_aligned[i] and
                  chop_range_aligned[i]):
                position = -1
                entry_price = high[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = -0.25
    
    return signals