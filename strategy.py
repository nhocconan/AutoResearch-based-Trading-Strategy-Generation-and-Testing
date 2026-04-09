#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (from 1d) + volume spike + chop regime filter
# Camarilla levels provide intraday support/resistance from prior day's range
# Volume spike confirms institutional participation at these key levels
# Chop regime filter (CHOP > 61.8) ensures we only trade in ranging markets where mean reversion works
# In trending markets (CHOP < 38.2), we avoid false breakouts at Camarilla levels
# Works in bull/bear: chop filter adapts to market regime, Camarilla provides structure in any environment
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
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
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d OHLC
    # Camarilla: based on prior day's range
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.25 * (High - Low)
    # L3 = Close - 1.25 * (High - Low)
    # We use H3/L3 as primary entry/exit levels
    
    prior_high = df_1d['high'].values
    prior_low = df_1d['low'].values
    prior_close = df_1d['close'].values
    
    # Calculate Camarilla H3 and L3 levels
    camarilla_h3 = prior_close + 1.25 * (prior_high - prior_low)
    camarilla_l3 = prior_close - 1.25 * (prior_high - prior_low)
    
    # Align Camarilla levels to 12h timeframe (wait for 1d close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 20-period average volume for volume confirmation
    vol_series = pd.Series(volume)
    avg_volume = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index regime filter (14-period)
    # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (avoid)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # First bar TR
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop = np.full(n, 50.0)  # Default to neutral
    mask = (hh14 - ll14) > 0
    chop[mask] = 100 * np.log10(atr14[mask] * np.sqrt(14) / (hh14[mask] - ll14[mask])) / np.log10(10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 (mean reversion target reached) OR chop regime shifts to trending
            if close[i] < camarilla_l3_aligned[i] or chop[i] < 50.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 (mean reversion target reached) OR chop regime shifts to trending
            if close[i] > camarilla_h3_aligned[i] or chop[i] < 50.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and chop filter
            if volume_confirmed and chop_filter:
                # Long entry: price < Camarilla L3 (oversold bounce)
                if close[i] < camarilla_l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price > Camarilla H3 (overbought reversal)
                elif close[i] > camarilla_h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals