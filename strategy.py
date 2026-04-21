#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Volume_Chop_Regime_ATRStop_V1
Hypothesis: Camarilla R1/S1 breakout with volume confirmation (>1.3x average) and choppy market filter (Choppiness Index > 61.8) 
captures mean-reversion bounces in ranging markets while avoiding strong trends. Uses 1d HTF for Camarilla calculation and 
ATR(14) trailing stop (2.0*ATR) for risk control. Designed for low trade frequency (target: 20-40 trades/year) to minimize 
fee drag and work in both bull/bear markets via regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivot levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (based on previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day: use same day's data (will be refined as more data comes)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (already delayed by 1 day via roll)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h Indicators (primary timeframe) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * vol_ma
    
    # Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (n * (HHV - LLV))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = 100 * np.log10(sum_atr14 / (14 * (hh14 - ll14))) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((hh14 - ll14) == 0, 50, chop)  # Neutral when no range
    chop = np.where(np.isnan(chop), 50, chop)
    
    # ATR (14-period) for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(volume_threshold[i]) or np.isnan(chop[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + choppy market (mean reversion regime)
            if price > r1_aligned[i] and volume[i] > volume_threshold[i] and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume confirmation + choppy market (mean reversion regime)
            elif price < s1_aligned[i] and volume[i] > volume_threshold[i] and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below R1 (breakout failed)
            elif price < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above S1 (breakout failed)
            elif price > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Volume_Chop_Regime_ATRStop_V1"
timeframe = "4h"
leverage = 1.0