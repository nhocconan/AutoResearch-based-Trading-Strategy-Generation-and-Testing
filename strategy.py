#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1d chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average AND 1d chop < 38.2 (trending market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average AND 1d chop < 38.2 (trending market)
# - Exit when price returns to Camarilla Pivot level (mean reversion to equilibrium)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla levels from 1d provide institutional support/resistance; volume confirms breakout strength
# - Chop filter ensures we only trade in trending markets where breakouts work best
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: H4, H3, L3, L4, Pivot
    camarilla_h4 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_h3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_l3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_l4 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_pivot = np.full_like(close_1d, np.nan, dtype=float)
    
    for i in range(len(close_1d)):
        if i == 0:
            continue  # Need previous day data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        camarilla_pivot[i] = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        camarilla_h4[i] = prev_close + range_val * 1.1 / 2
        camarilla_h3[i] = prev_close + range_val * 1.1 / 4
        camarilla_l3[i] = prev_close - range_val * 1.1 / 4
        camarilla_l4[i] = prev_close - range_val * 1.1 / 2
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 1d Choppiness Index (14-period)
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.zeros_like(tr_1d)
    for i in range(14, len(tr_1d)):
        atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    hh_1d = np.full_like(close_1d, np.nan, dtype=float)
    ll_1d = np.full_like(close_1d, np.nan, dtype=float)
    for i in range(13, len(close_1d)):
        hh_1d[i] = np.max(high_1d[i-13:i+1])
        ll_1d[i] = np.min(low_1d[i-13:i+1])
    
    chop_1d = np.full_like(close_1d, np.nan, dtype=float)
    for i in range(13, len(close_1d)):
        if hh_1d[i] > ll_1d[i]:
            tr_sum = np.sum(tr_1d[i-13:i+1])
            chop_1d[i] = 100 * np.log10(tr_sum / (hh_1d[i] - ll_1d[i])) / np.log10(14)
        else:
            chop_1d[i] = 50.0
    
    chop_regime_1d = chop_1d < 38.2  # Trending market (chop < 38.2)
    
    # Align HTF indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(chop_regime_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND trending regime
            if close[i] > camarilla_h3_aligned[i] and volume[i] > 1.5 * vol_ma_1d_aligned[i] and chop_regime_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND trending regime
            elif close[i] < camarilla_l3_aligned[i] and volume[i] > 1.5 * vol_ma_1d_aligned[i] and chop_regime_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Pivot level (mean reversion)
            exit_long = position == 1 and close[i] <= camarilla_pivot_aligned[i]
            exit_short = position == -1 and close[i] >= camarilla_pivot_aligned[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals