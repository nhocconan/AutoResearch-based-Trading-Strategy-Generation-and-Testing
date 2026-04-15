#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot (R1/S1) breakout with volume confirmation and 1w EMA200 trend filter.
# Uses 1d Camarilla levels for institutional structure, filtered by weekly trend to avoid counter-trend trades.
# Volume confirmation ensures institutional participation. Designed for low trade frequency (20-40/year) to minimize fee drag.
# Works in bull/bear: 1w EMA200 defines primary trend, Camarilla breakouts capture significant level reactions.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivots (R1, S1) ===
    # Camarilla formula: Close +- (High-Low) * 1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_high = close_1d + (high_1d - low_1d) * 1.1 / 12  # R1 level
    camarilla_low = close_1d - (high_1d - low_1d) * 1.1 / 12   # S1 level
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(200) for primary trend bias
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 50-period volume SMA
        vol_sma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
        vol_confirm = volume[i] > (vol_sma_50[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1 (resistance)
        # 2. 1w price above EMA200 (bullish primary trend)
        # 3. Volume confirmation
        if (close[i] > camarilla_high_aligned[i] and
            close[i] > ema_200_1w_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1 (support)
        # 2. 1w price below EMA200 (bearish primary trend)
        # 3. Volume confirmation
        elif (close[i] < camarilla_low_aligned[i] and
              close[i] < ema_200_1w_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_1d_Camarilla_R1S1_1w_EMA200_VolFilter"
timeframe = "4h"
leverage = 1.0