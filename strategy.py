#!/usr/bin/env python3
"""
4h_KAMA_Trend_Regime_Volume_v1
Hypothesis: 4h KAMA trend direction with choppiness regime filter and volume confirmation.
KAMA adapts to market noise - trending when ER high, mean-reverting when ER low.
Only trade in direction of KAMA trend when market is trending (CHOP < 40) or mean-reverting (CHOP > 60) with volume spike.
Uses discrete sizing (0.25) to minimize fees. Designed for 20-50 trades/year on 4h.
Works in bull/bear via regime adaptation: trend follow in trending markets, mean revert in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF regime and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for trend direction
    # Efficiency Ratio (ER) = |Change| / Sum|Changes|
    change = abs(df_1d['close'].diff(10))
    volatility = df_1d['close'].diff().abs().rolling(10).sum()
    ER = change / volatility.replace(0, np.nan)
    # Smoothing Constants: fastest EMA=2, slowest EMA=30
    SC = (ER * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(len(df_1d))
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        if np.isnan(SC.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + SC.iloc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    # KAMA trend: 1 if price > KAMA, -1 if price < KAMA
    kama_trend = np.where(close > kama_aligned, 1, -1)
    
    # Calculate 1d Choppiness Index for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    atr_1d = np.zeros(len(df_1d))
    tr_1d = np.maximum(df_1d['high'], df_1d['close'].shift(1)) - np.minimum(df_1d['low'], df_1d['close'].shift(1))
    tr_1d.iloc[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_1d = pd.Series(tr_1d).rolling(14).mean().values
    
    max_high_1d = pd.Series(df_1d['high']).rolling(14).max().values
    min_low_1d = pd.Series(df_1d['low']).rolling(14).min().values
    chop_denominator = max_high_1d - min_low_1d
    chop_denominator = np.where(chop_denominator == 0, 1, chop_denominator)  # avoid div by zero
    chop_value = 100 * np.log10(pd.Series(atr_1d).rolling(14).sum() / chop_denominator) / np.log10(14)
    chop_value = np.nan_to_num(chop_value, nan=50.0)  # fill NaN with neutral
    
    # Align Choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_value)
    
    # Calculate 20-period volume average for spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime-based logic
        if chop_aligned[i] < 40:  # Trending regime
            # Follow KAMA trend
            if kama_trend[i] == 1 and volume_spike:  # Uptrend
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif kama_trend[i] == -1 and volume_spike:  # Downtrend
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif chop_aligned[i] > 60:  # Ranging regime
            # Mean reversion: fade moves away from KAMA
            if close[i] < kama_aligned[i] and volume_spike:  # Price below KAMA -> long
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] > kama_aligned[i] and volume_spike:  # Price above KAMA -> short
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:  # Neutral regime (40 <= CHOP <= 60)
            # No clear edge, stay flat or reduce position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.10  # Reduce long exposure
            else:
                signals[i] = -0.10  # Reduce short exposure
    
    return signals

name = "4h_KAMA_Trend_Regime_Volume_v1"
timeframe = "4h"
leverage = 1.0