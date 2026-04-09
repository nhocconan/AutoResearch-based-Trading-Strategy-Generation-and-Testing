#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume confirmation + choppiness regime filter
# Long when price touches Camarilla L3 support with volume spike in choppy market (mean reversion)
# Short when price touches Camarilla H3 resistance with volume spike in choppy market
# Uses discrete position sizing 0.25 to target ~25-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion in chop, volume confirms institutional interest

name = "4h_1d_camarilla_pivot_volume_chop_v1"
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
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    #          L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Calculate 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    vol_s_1d = pd.Series(vol_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 4h choppiness index (14-period) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close_4h = np.roll(close, 1)
    prev_close_4h[0] = np.nan
    tr = true_range(high, low, prev_close_4h)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(atr14) / (max(high)-min(low))) / log10(14)
    # Simplified: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop[max_high_14 == min_low_14] = 50  # Avoid division by zero
    
    # Align chop to ensure proper timing (already on 4h)
    chop_aligned = chop  # No need to align as it's already calculated on 4h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x average 1h volume (approximated)
        # Since we don't have 1h volume aligned, use 4h volume vs its own 20-period MA
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Regime filter: only trade in choppy market (Chop > 61.8) for mean reversion
        in_choppy_regime = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long if price rises above Camarilla H3 (profit target) or falls below L3 (stop)
            if close[i] > camarilla_h3_aligned[i] or close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price falls below Camarilla L3 (profit target) or rises above H3 (stop)
            if close[i] < camarilla_l3_aligned[i] or close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion strategy: enter at Camarilla L3/H3 with volume confirmation in choppy market
            if close[i] <= camarilla_l3_aligned[i] and volume_confirmed and in_choppy_regime:
                position = 1
                signals[i] = 0.25
            elif close[i] >= camarilla_h3_aligned[i] and volume_confirmed and in_choppy_regime:
                position = -1
                signals[i] = -0.25
    
    return signals