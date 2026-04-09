#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and chop regime filter
# Uses 1d Camarilla levels (H3/L3) for structure, enters on break with 1.5x volume spike
# Only trades when market is trending (CHOP < 38.2) to avoid false breakouts in ranging markets
# Discrete position sizing 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends, chop filter avoids whipsaws in ranges

name = "4h_1d_camarilla_breakout_v26"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #            L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    rng = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * rng * 1.1 / 2
    camarilla_h3 = close_1d + 1.1 * rng * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * rng * 1.1 / 4
    camarilla_l4 = close_1d - 1.1 * rng * 1.1 / 2
    
    # Calculate 1d average volume (20-period)
    vol_s_1d = pd.Series(df_1d['volume'].values)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1h chopiness index (14-period) for regime filter
    def true_range(high_arr, low_arr, close_arr):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first bar
        return tr
    
    tr = true_range(high, low, close)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum(tr[-14:]) / (highest_high_14[-1] - lowest_low_14[-1])) if len(tr) >= 14 else 50
    # Vectorized chop calculation
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = highest_high_14 - lowest_low_14
    chop = 100 * np.log10(sum_tr_14 / hh_ll_diff)
    chop[hh_ll_diff == 0] = 50  # avoid division by zero
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 4h volume average (20-period) for confirmation
    vol_s_4h = pd.Series(volume)
    avg_vol_4h = vol_s_4h.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_4h[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * avg_vol_4h[i] if not np.isnan(avg_vol_4h[i]) else False
        
        # Regime filter: only trade when trending (CHOP < 38.2)
        trending_market = chop_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit long if price falls below Camarilla L3
            if close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above Camarilla H3
            if close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on Camarilla H3/L3 breakout with volume confirmation in trending market
            if volume_confirmed and trending_market:
                if close[i] > camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals