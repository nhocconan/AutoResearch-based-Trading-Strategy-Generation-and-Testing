#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike + chop regime filter
    # Uses 1d Camarilla levels (H3/L3) for structure: long above H3, short below L3
    # Volume confirmation: volume > 2.0 * 20-period average to filter false breakouts
    # Chop regime: only trade when market is trending (CHOP < 38.2) to avoid whipsaws in ranging markets
    # Discrete sizing 0.25 to minimize fee churn. Target: 20-30 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (H3, L3) - using previous day's range
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        # Calculate based on previous day's high-low-close
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        range_val = phigh - plow
        camarilla_h3[i] = pclose + range_val * 1.1 / 4
        camarilla_l3[i] = pclose - range_val * 1.1 / 4
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_1d = np.full(len(close_1d), np.nan)
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = np.abs(high_1d[i] - close_1d[i-1])
        lc = np.abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(hl, hc, lc)
    
    for i in range(14, len(close_1d)):
        atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    chop_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        sum_atr = np.sum(atr_1d[i-13:i+1])
        if highest_high != lowest_low:
            chop_1d[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(highest_high - lowest_low)
        else:
            chop_1d[i] = 0
    
    # Align HTF indicators to LTF
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_1d_aligned[i] < 38.2
        
        # Entry logic: Camarilla breakout with volume and regime filter
        long_entry = False
        short_entry = False
        
        if is_trending:
            # Long breakout: price breaks above Camarilla H3 with volume spike
            long_entry = (close[i] > camarilla_h3_aligned[i]) and volume_spike[i]
            # Short breakout: price breaks below Camarilla L3 with volume spike
            short_entry = (close[i] < camarilla_l3_aligned[i]) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or chop regime shift to ranging
        long_exit = (not is_trending) or (close[i] < camarilla_l3_aligned[i])
        short_exit = (not is_trending) or (close[i] > camarilla_h3_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0