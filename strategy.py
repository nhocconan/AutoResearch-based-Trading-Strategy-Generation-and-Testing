#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA trend filter and volume confirmation.
# Uses 12h EMA(50) for intermediate trend bias and 6h Elder Ray for momentum assessment.
# Long when Bull Power > 0 and price above 12h EMA50 with volume confirmation.
# Short when Bear Power < 0 and price below 12h EMA50 with volume confirmation.
# Designed for low trade frequency (12-37/year) to minimize fee drag while capturing
# momentum shifts in both trending and ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h and 12h HTF data once before loop
    df_6h = get_htf_data(prices, '6h')
    df_12h = get_htf_data(prices, '12h')
    if len(df_6h) < 50 or len(df_12h) < 50:
        return np.zeros(n)
    
    # === 6h Indicators: Elder Ray Index ===
    # EMA13 for 6h (standard for Elder Ray)
    ema13_6h = pd.Series(df_6h['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_6h['high'].values - ema13_6h  # Bull Power = High - EMA13
    bear_power = df_6h['low'].values - ema13_6h   # Bear Power = Low - EMA13
    
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # === 12h Indicators: Trend Filter ===
    # 12h EMA(50) for intermediate trend bias
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power positive (buying pressure)
        # 2. Price above 12h EMA50 (bullish intermediate trend)
        # 3. Volume confirmation
        if (bull_power_aligned[i] > 0 and
            close[i] > ema50_12h_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power negative (selling pressure)
        # 2. Price below 12h EMA50 (bearish intermediate trend)
        # 3. Volume confirmation
        elif (bear_power_aligned[i] < 0 and
              close[i] < ema50_12h_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_EMA50_VolFilter_v1"
timeframe = "6h"
leverage = 1.0