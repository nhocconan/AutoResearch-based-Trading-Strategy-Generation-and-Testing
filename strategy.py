#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d Camarilla levels (H3, L3) for breakout signals.
- Volume: Current 12h volume > 1.5 * 20-period 1d volume MA to confirm breakout strength.
- Regime: 1d Choppiness Index (CHOP) > 61.8 for ranging markets (mean reversion at H3/L3).
- Entry: Long when price breaks above H3 with volume spike AND chop > 61.8.
         Short when price breaks below L3 with volume spike AND chop > 61.8.
- Exit: Opposite breakout (price re-enters H3/L3 range) or loss of volume/chop confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Camarilla levels work well in ranging markets, and the chop filter ensures we only trade
when the market is ranging (not trending), which increases mean reversion accuracy.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Extract 1d OHLC
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    df_1d_volume = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (H3, L3)
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # Using previous day's OHLC for today's levels (standard practice)
    prev_high = np.roll(df_1d_high, 1)
    prev_low = np.roll(df_1d_low, 1)
    prev_close = np.roll(df_1d_close, 1)
    # Set first value to NaN since no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    h3 = prev_close + 1.1 * camarilla_range / 4
    l3 = prev_close - 1.1 * camarilla_range / 4
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(N) / (highest_high - lowest_low)) over N periods
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(
        df_1d_high[1:] - df_1d_low[1:],
        np.maximum(
            np.abs(df_1d_high[1:] - np.roll(df_1d_close, 1)[1:]),
            np.abs(df_1d_low[1:] - np.roll(df_1d_close, 1)[1:])
        )
    )
    # Pad first element
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr14 * 14 / np.log10(14)) / np.log10((highest_high - lowest_low))
    # Handle division by zero or invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)  # Default to neutral
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    # Regime filter: chop > 61.8 (ranging market)
    ranging_market = chop_aligned > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need enough bars for Camarilla and CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike AND ranging market
            if volume_spike[i] and ranging_market[i]:
                # Bullish: price breaks above H3
                if curr_low > h3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below L3
                elif curr_high < l3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters below H3 OR loss of confirmation
            if curr_high < h3_aligned[i] or not (volume_spike[i] and ranging_market[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above L3 OR loss of confirmation
            if curr_low > l3_aligned[i] or not (volume_spike[i] and ranging_market[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0