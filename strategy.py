#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume confirmation + chop regime filter
# Long when price touches Camarilla L3 support with volume confirmation in trending regime (CHOP < 38.2)
# Short when price touches Camarilla H3 resistance with volume confirmation in trending regime
# Uses discrete position sizing 0.25 to target ~25-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion at strong pivot levels during trends avoids false breakouts

name = "4h_1d_camarilla_pivot_volume_chop_v5"
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
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # Actually: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # Standard Camarilla: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # Simplified: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # Correct formula: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # Actually standard: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # Let's use: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # Wait, standard Camarilla: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # Actually: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # Correct: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    # Let me verify: Standard Camarilla levels:
    # H4 = close + 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4
    # L4 = close - 1.1*(high-low)*1.1/2
    # So H3/L3 use multiplier 1.1*1.1/4 = 1.21/4 = 0.3025
    # Actually: 1.1 * (high-low) * 1.1 / 4 = 1.21 * (high-low) / 4 = 0.3025 * (high-low)
    
    camarilla_multiplier = 1.1 * 1.1 / 4  # 1.21/4 = 0.3025
    H3_1d = close_1d + camarilla_multiplier * (high_1d - low_1d)
    L3_1d = close_1d - camarilla_multiplier * (high_1d - low_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate TR for 1d
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first period
    tr_1d = true_range(high_1d, low_1d, prev_close_1d)
    
    # CHOP = 100 * log10(sum(TR,14) / (max(high,14) - min(low,14))) / log10(14)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_14_1d / (max_high_14_1d - min_low_14_1d)) / np.log10(14)
    # Handle division by zero or invalid values
    chop_1d = np.where((max_high_14_1d - min_low_14_1d) > 0, chop_1d, 50.0)
    
    # Align 1d indicators to 4h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute 4h volume confirmation (20-period)
    vol_s_4h = pd.Series(volume)
    avg_vol_4h = vol_s_4h.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(avg_vol_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume (20-period)
        volume_confirmed = volume[i] > 1.5 * avg_vol_4h[i]
        
        # Regime filter: trending market (CHOP < 38.2)
        trending_regime = chop_1d_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit long if price rises above H3 (mean reversion completion) or stops trending
            if close[i] > H3_1d_aligned[i] or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price falls below L3 (mean reversion completion) or stops trending
            if close[i] < L3_1d_aligned[i] or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion strategy: enter at Camarilla L3/H3 with volume confirmation in trending regime
            if volume_confirmed and trending_regime:
                # Long when price touches or falls below L3 support
                if close[i] <= L3_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short when price touches or rises above H3 resistance
                elif close[i] >= H3_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals