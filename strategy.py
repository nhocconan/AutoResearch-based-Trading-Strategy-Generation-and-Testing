#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop filter
# - Long when price breaks above Camarilla H3 (1d) with 12h volume > 1.5x 20-period average and CHOP(14) < 38.2 (trending)
# - Short when price breaks below Camarilla L3 (1d) with same filters
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Designed for 12h timeframe: targets 12-37 trades/year to avoid fee drag
# - Works in bull/bear markets: CHOP filter ensures we trade only in trending conditions, avoiding whipsaws in ranging markets
# - Camarilla levels from 1d provide strong intraday support/resistance levels

name = "12h_1d_camarilla_volume_chop_v1"
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
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla pivot levels for each day
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.5 * avg_volume_20)
    
    # Pre-compute 12h Choppiness Index (CHOP) for regime filter
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATM (Average True Range) over 14 periods
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Maximum and minimum close over 14 periods
    max_close_14 = pd.Series(close_12h).rolling(window=14, min_periods=14).max().values
    min_close_14 = pd.Series(close_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(atr_14) / (max_close_14 - min_close_14)) / log10(14)
    # Avoid division by zero
    range_14 = max_close_14 - min_close_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    chop[np.isnan(chop)] = 50  # Set neutral value when undefined
    
    # Trending market: CHOP < 38.2
    trending_market = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(trending_market[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price re-enters below Camarilla H3 (failed breakout)
            if close_12h[i] < camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters above Camarilla L3 (failed breakout)
            if close_12h[i] > camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume and trend filters
            if vol_spike[i] and trending_market[i]:
                # Breakout long: price closes above Camarilla H3
                if close_12h[i] > camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price closes below Camarilla L3
                elif close_12h[i] < camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals