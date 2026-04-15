#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Uses 1d EMA50 for trend bias and Williams Alligator (SMAs with specific periods/offsets) on 12h for entry timing.
# Includes volume filter (current volume > 1.5x 20-bar SMA) to avoid low-momentum breakouts.
# Designed for low trade frequency (15-30/year) to minimize fee drag in choppy markets.
# Works in bull/bear: Alligator identifies trend direction, volume confirms momentum, 1d EMA avoids counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 12h and 1d HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 12h Indicators: Williams Alligator ===
    # Williams Alligator: Jaw (13-period SMA, 8-bar offset), Teeth (8-period SMA, 5-bar offset), Lips (5-period SMA, 3-bar offset)
    close_12h = pd.Series(df_12h['close'].values)
    jaw_12h = close_12h.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_12h = close_12h.rolling(window=8, min_periods=8).mean().shift(5).values
    lips_12h = close_12h.rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current 12h volume > 1.5x 20-period 12h volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Lips > Teeth > Jaw (Alligator bullish alignment: mouth opening upward)
        # 2. Price above Lips (confirming upward momentum)
        # 3. 1d price above EMA50 (bullish long-term trend bias)
        # 4. Volume confirmation
        if (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i] and
            close[i] > lips_12h_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Jaw > Teeth > Lips (Alligator bearish alignment: mouth opening downward)
        # 2. Price below Jaw (confirming downward momentum)
        # 3. 1d price below EMA50 (bearish long-term trend bias)
        # 4. Volume confirmation
        elif (jaw_12h_aligned[i] > teeth_12h_aligned[i] > lips_12h_aligned[i] and
              close[i] < jaw_12h_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Williams_Alligator_EMA50_VolFilter_v1"
timeframe = "12h"
leverage = 1.0