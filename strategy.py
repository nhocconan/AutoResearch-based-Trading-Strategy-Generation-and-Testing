#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume confirmation + choppiness regime filter
# Long when price breaks above 1d Camarilla H3 with volume confirmation in trending market (CHOP < 38.2)
# Short when price breaks below 1d Camarilla L3 with volume confirmation in trending market
# Uses discrete position sizing 0.25 to target ~25-40 trades/year and minimize fee drag
# Works in bull/bear markets: Camarilla levels provide adaptive support/resistance, volume confirms breakout strength,
# choppiness filter avoids false signals in ranging markets

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
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4
    # Using previous day's values to avoid look-ahead
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan  # First day has no previous
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_h3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 4
    camarilla_l3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # Calculate 14-period choppiness index on 1d timeframe
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close_1d_tr = np.roll(close_1d, 1)
    prev_close_1d_tr[0] = np.nan
    tr_1d = true_range(high_1d, low_1d, prev_close_1d_tr)
    
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    highest_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14)/(log10(14)* (HH14-LL14)))
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    denominator = np.log10(14) * (highest_high_14_1d - lowest_low_14_1d)
    chop_1d = 100 * np.log10(sum_tr_14 / denominator)
    # Handle division by zero or invalid values
    chop_1d = np.where((highest_high_14_1d - lowest_low_14_1d) > 0, chop_1d, 50.0)
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current 4h volume > 1.5x average 4h volume (20-period)
    vol_s_4h = pd.Series(volume)
    avg_vol_4h = vol_s_4h.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(avg_vol_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * avg_vol_4h[i]
        
        # Trending market filter: CHOP < 38.2 (trending), avoid ranging markets (CHOP > 61.8)
        trending_market = chop_1d_aligned[i] < 38.2
        
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
            # Breakout strategy: enter on Camarilla breakout with volume confirmation in trending market
            if close[i] > camarilla_h3_aligned[i] and volume_confirmed and trending_market:
                position = 1
                signals[i] = 0.25
            elif close[i] < camarilla_l3_aligned[i] and volume_confirmed and trending_market:
                position = -1
                signals[i] = -0.25
    
    return signals