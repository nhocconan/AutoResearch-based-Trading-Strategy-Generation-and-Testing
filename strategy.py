#!/usr/bin/env python3
# 12h_camarilla_1d_volume_chop_v2
# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and choppiness regime filter.
# Enters long when price breaks above H3 with volume > 1.5x average and chop < 61.8 (trending).
# Enters short when price breaks below L3 with volume > 1.5x average and chop < 61.8 (trending).
# Uses discrete sizing (±0.25) to minimize fee churn. Designed for low trade frequency (<400 total trades).
# Works in bull/bear by using 1d pivot levels as dynamic support/resistance and chop filter to avoid ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_volume_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d HTF data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d (using previous day's OHLC)
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.5 * (high - low)
    # H1 = close + 0.25 * (high - low)
    # L1 = close - 0.25 * (high - low)
    # L2 = close - 0.5 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    rng = high_1d - low_1d
    h3 = close_1d + 1.0 * rng
    l3 = close_1d - 1.0 * rng
    
    # Align 1d Camarilla levels to 12h timeframe (completed 1d candle only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # 1d HTF data for choppiness regime filter
    # Chop = 100 * log10(sum(ATR,14) / (log10(highest-highest-lowest-lowest,14))) / log10(14)
    # But we'll use a simplified version: Chop = 100 * ATR(14) / (HHV(14) - LLV(14))
    # Chop > 61.8 = ranging, Chop < 38.2 = trending
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * (atr_1d / (hh_1d - ll_1d + 1e-10))
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 12h indicators for volume confirmation
    # Volume ratio: current volume / average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (chop < 61.8)
        trending = chop_1d_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price falls back below H3
            if close[i] < h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises back above L3
            if close[i] > l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with volume confirmation
            if (close[i] > h3_aligned[i]) and \
               (vol_ratio[i] > 1.5) and \
               trending:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volume confirmation
            elif (close[i] < l3_aligned[i]) and \
                 (vol_ratio[i] > 1.5) and \
                 trending:
                position = -1
                signals[i] = -0.25
    
    return signals