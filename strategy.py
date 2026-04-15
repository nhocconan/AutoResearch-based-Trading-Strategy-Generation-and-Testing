#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with volume spike and chop regime filter.
# Works in bull/bear: Camarilla levels act as support/resistance, volume confirms breakout,
# chop filter avoids whipsaws in ranging markets. Target: 12-37 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R1, S1, R4, S4)
    # Using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    camarilla_r1 = prev_close + prev_range * 1.1 / 12
    camarilla_s1 = prev_close - prev_range * 1.1 / 12
    camarilla_r4 = prev_close + prev_range * 1.1 / 2
    camarilla_s4 = prev_close - prev_range * 1.1 / 2
    
    # Align to 12h timeframe (wait for completed daily candle)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume spike: 24h volume > 1.5 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma_20
    
    # Choppiness regime filter: avoid strong trends (CHOP > 50 = choppy/range)
    # Using 14-period chop on 12h data
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14.sum() / np.log(14) / (highest_high_14 - lowest_low_14))
    chop[~np.isfinite(chop)] = 50  # default to neutral when invalid
    chop_filter = chop > 50  # trade in choppy/range conditions
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price breaks above Camarilla R1 (resistance)
        # 2. Volume spike confirms breakout
        # 3. Choppy market (avoid strong trends)
        if (close[i] > camarilla_r1_aligned[i] and
            vol_spike[i] and
            chop_filter[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below Camarilla S1 (support)
        # 2. Volume spike confirms breakdown
        # 3. Choppy market (avoid strong trends)
        elif (close[i] < camarilla_s1_aligned[i] and
              vol_spike[i] and
              chop_filter[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Volume_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0