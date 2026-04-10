#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 12h volume spike + 1w choppiness regime
# - Primary: 4h price breaking above/below Camarilla H3/L3 levels from prior day
# - Volume filter: 12h volume > 1.8x 20-period volume MA to confirm breakout strength
# - Regime filter: 1w choppiness index > 61.8 (range market) for mean reversion exits
# - Exit: Price returns to Camarilla pivot point (mean reversion in chop)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla adapts to volatility, volume confirms genuine breakouts,
#   chop filter avoids whipsaws in ranging markets
# - Target: 80-160 total trades over 4 years = 20-40/year for 4h timeframe

name = "4h_12h_1w_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    if len(df_12h) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate prior day's OHLC for Camarilla (using 1d data aligned to 4h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4.0
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4.0
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 12h volume confirmation: volume > 1.8x 20-period volume MA
    volume_ma_20_12h = pd.Series(volume_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Calculate 1w choppiness index: CHOP = 100 * log10(sum(ATR(14)) / log10(14 * (HH - LL)))
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14_sum / np.log10(14 * (hh14 - ll14)))
    chop_raw = np.where((hh14 - ll14) > 0, chop_raw, 50.0)
    chop_1w = chop_raw
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_ma_20_12h_aligned[i]) or 
            np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.8x 20-period MA
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)
        vol_confirm = vol_12h_current[i] > 1.8 * volume_ma_20_12h_aligned[i]
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        regime_filter = chop_1w_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries at Camarilla breakouts
            # Long entry: price breaks above H3 + vol confirmation + chop regime
            if close[i] > camarilla_h3_aligned[i] and vol_confirm and regime_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L3 + vol confirmation + chop regime
            elif close[i] < camarilla_l3_aligned[i] and vol_confirm and regime_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point (mean reversion)
            # Exit: price returns to Camarilla pivot level
            if position == 1:  # Long position
                if close[i] <= camarilla_pivot_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= camarilla_pivot_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals