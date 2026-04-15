#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA50 trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Goes long when Bull Power > 0 AND rising (momentum) AND 1d trend bullish AND volume confirms.
# Goes short when Bear Power < 0 AND falling (momentum) AND 1d trend bearish AND volume confirms.
# Designed for low trade frequency (12-30/year) to minimize fee drag. Works in bull/bear: 1d EMA avoids counter-trend trades,
# Elder Ray captures momentum shifts with EMA13 smoothing.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h and 1d HTF data once before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_6h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 6h Indicators: Elder Ray ===
    # EMA13 for Elder Ray base
    close_6h = pd.Series(df_6h['close'].values)
    ema13_6h = close_6h.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_6h = pd.Series(df_6h['high'].values)
    low_6h = pd.Series(df_6h['low'].values)
    bull_power_6h = (high_6h - ema13_6h).values
    bear_power_6h = (low_6h - ema13_6h).values
    
    # Slope of Bull/Bear Power (momentum): 3-bar change
    bull_power_slope = pd.Series(bull_power_6h).diff(3).values
    bear_power_slope = pd.Series(bear_power_6h).diff(3).values
    
    # Align to LTF
    ema13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema13_6h)
    bull_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    bull_power_slope_aligned = align_htf_to_ltf(prices, df_6h, bull_power_slope)
    bear_power_slope_aligned = align_htf_to_ltf(prices, df_6h, bear_power_slope)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current 6h volume > 1.5x 20-period 6h volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema13_6h_aligned[i]) or np.isnan(bull_power_6h_aligned[i]) or
            np.isnan(bear_power_6h_aligned[i]) or np.isnan(bull_power_slope_aligned[i]) or
            np.isnan(bear_power_slope_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (price above EMA13 = bullish momentum)
        # 2. Bull Power rising (accelerating bullish momentum)
        # 3. 6h close above EMA13 (additional confirmation)
        # 4. 1d price above EMA50 (bullish long-term trend bias)
        # 5. Volume confirmation
        if (bull_power_6h_aligned[i] > 0 and
            bull_power_slope_aligned[i] > 0 and
            close[i] > ema13_6h_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (price below EMA13 = bearish momentum)
        # 2. Bear Power falling (accelerating bearish momentum)
        # 3. 6h close below EMA13 (additional confirmation)
        # 4. 1d price below EMA50 (bearish long-term trend bias)
        # 5. Volume confirmation
        elif (bear_power_6h_aligned[i] < 0 and
              bear_power_slope_aligned[i] < 0 and
              close[i] < ema13_6h_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Elder_Ray_EMA50_VolFilter_v1"
timeframe = "6h"
leverage = 1.0