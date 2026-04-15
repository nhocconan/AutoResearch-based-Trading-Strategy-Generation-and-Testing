#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator strategy with 1w trend filter and volume confirmation.
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trending vs ranging markets.
# Enters long when Lips > Teeth > Jaw (bullish alignment) and price > 1w EMA34.
# Enters short when Lips < Teeth < Jaw (bearish alignment) and price < 1w EMA34.
# Includes volume filter (current volume > 1.3x 20-bar SMA) to avoid low-momentum entries.
# Designed for very low trade frequency (12-37/year) to minimize fee drag in choppy markets.
# Williams Alligator is effective in both bull and bear markets by filtering out sideways action.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h and 1w HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    if len(df_12h) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 12h Indicators: Williams Alligator ===
    median_price = (df_12h['high'].values + df_12h['low'].values) / 2.0
    close_series = pd.Series(df_12h['close'].values)
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = close_series.rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = close_series.rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, shifted 3 bars
    lips = close_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(34) for long-term trend bias
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish alignment: Lips > Teeth > Jaw
        # 2. Price above 1w EMA34 (bullish long-term trend bias)
        # 3. Volume confirmation
        if (lips_aligned[i] > teeth_aligned[i] and
            teeth_aligned[i] > jaw_aligned[i] and
            close[i] > ema_34_1w_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish alignment: Lips < Teeth < Jaw
        # 2. Price below 1w EMA34 (bearish long-term trend bias)
        # 3. Volume confirmation
        elif (lips_aligned[i] < teeth_aligned[i] and
              teeth_aligned[i] < jaw_aligned[i] and
              close[i] < ema_34_1w_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsAlligator_EMA34_VolFilter_v1"
timeframe = "12h"
leverage = 1.0