#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels from prior 1d
# - Volume filter: 1d volume > 1.3x 20-period average volume (institutional participation)
# - Regime filter: 1d Choppiness Index > 61.8 (range market) for mean reversion at H3/L3
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 4h
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Camarilla levels act as support/resistance; volume confirms institutional interest;
#   chop filter ensures we're in ranging conditions where mean reversion at pivots works

name = "4h_1d_camarilla_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.3 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1 = high_1d - low_1d
    tr_2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    tr[0] = tr_1[0]
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(tr) / (hh - ll)) / log10(14)
    # Avoid division by zero
    hh_ll = hh - ll
    chop = np.where(hh_ll > 0, 100 * np.log10(tr_sum / hh_ll) / np.log10(14), 50)
    chop_filter = chop > 61.8  # Chop > 61.8 = ranging market
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # Pre-compute Camarilla levels from prior 1d (H3, L3, H4, L4)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #            L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    hl_range = df_1d['high'].values - df_1d['low'].values
    camarilla_h3 = df_1d['close'].values + (1.1 * hl_range * 1.1 / 4)
    camarilla_l3 = df_1d['close'].values - (1.1 * hl_range * 1.1 / 4)
    camarilla_h4 = df_1d['close'].values + (1.1 * hl_range * 1.1 / 2)
    camarilla_l4 = df_1d['close'].values - (1.1 * hl_range * 1.1 / 2)
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 4h ATR(14) for stoploss
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_aligned[i]) or np.isnan(chop_filter_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price reaches H4 (profit target) OR stoploss hit OR chop breaks down (trend)
            if (close_4h[i] >= camarilla_h4_aligned[i] or 
                close_4h[i] <= entry_price - 1.5 * atr_14[i] or
                chop_filter_aligned[i] == False):  # Chop < 61.8 = trending, exit mean reversion
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reaches L4 (profit target) OR stoploss hit OR chop breaks down (trend)
            if (close_4h[i] <= camarilla_l4_aligned[i] or 
                close_4h[i] >= entry_price + 1.5 * atr_14[i] or
                chop_filter_aligned[i] == False):  # Chop < 61.8 = trending, exit mean reversion
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla H3/L3 breakouts with volume and chop filters
            if vol_spike_aligned[i] and chop_filter_aligned[i]:
                # Long: price breaks above H3 with conviction (close above H3)
                if close_4h[i] > camarilla_h3_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price breaks below L3 with conviction (close below L3)
                elif close_4h[i] < camarilla_l3_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals