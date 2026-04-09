#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume spike + choppiness regime filter
# Long when price touches/breaks above Camarilla H3 level with volume confirmation in low-chop regime
# Short when price touches/breaks below Camarilla L3 level with volume confirmation in low-chop regime
# Uses discrete position sizing 0.25 to target ~25-40 trades/year
# Camarilla pivots provide institutional support/resistance, volume confirms institutional interest,
# chop filter avoids false signals in ranging markets. Works in bull/bear via mean reversion at extremes.

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), 
    #            L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First bar has no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.0 * (prev_high - prev_low)
    
    # Calculate 1d average volume (20-period) for volume spike detection
    vol_1d = df_1d['volume'].values
    vol_s_1d = pd.Series(vol_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate choppiness index on 1d (14-period) for regime filter
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    tr = true_range(high_1d, low_1d, prev_close_1d)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(atr14) / (max(high) - min(low))) / log10(14)
    # But we use simplified version: high-low range vs ATR sum
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    chop = np.where(
        (range_14 > 0) & (~np.isnan(range_14)) & (~np.isnan(sum_atr_14)) & (sum_atr_14 > 0),
        100 * np.log10(sum_atr_14 / range_14) / np.log10(14),
        50.0  # neutral when invalid
    )
    # Regime: chop < 38.2 = trending, chop > 61.8 = ranging, 38.2-61.8 = transition
    # We want low chop (trending) for breakouts, but actually for Camarilla we want mean reversion
    # So we use chop > 50 as ranging regime (mean reversion works better in ranging markets)
    ranging_regime = chop > 50
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    ranging_regime_aligned = align_htf_to_ltf(prices, df_1d, ranging_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(ranging_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x average 1h volume (approximated)
        # Since we don't have 1h volume aligned, use 4h volume vs its own 20-period MA
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = not np.isnan(vol_ma_20[i]) and volume[i] > 2.0 * vol_ma_20[i]
        
        # Only trade in ranging regime (choppy market) where mean reversion at pivots works
        in_ranging = ranging_regime_aligned[i] > 0.5
        
        if position == 1:  # Long position
            # Exit long if price rises above H3 (took profit) or falls below L3 (stop)
            if close[i] > camarilla_h3_aligned[i] or close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price falls below L3 (took profit) or rises above H3 (stop)
            if close[i] < camarilla_l3_aligned[i] or close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion strategy: enter when price touches/pierces Camarilla levels
            # with volume confirmation in ranging market
            if volume_confirmed and in_ranging:
                if close[i] <= camarilla_l3_aligned[i]:  # Touched/below L3 -> long (mean reversion up)
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= camarilla_h3_aligned[i]:  # Touched/above H3 -> short (mean reversion down)
                    position = -1
                    signals[i] = -0.25
    
    return signals