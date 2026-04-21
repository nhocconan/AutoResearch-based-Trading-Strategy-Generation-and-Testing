#!/usr/bin/env python3
"""
1d_KAMA_Regime_Trend_ATRStop_v3
Hypothesis: Daily KAMA direction + chop regime filter + ATR-based trailing stop.
KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends.
Choppiness Index regime filter avoids trend-following in high-chop environments.
Designed for low trade frequency (<20 trades/year) to minimize fee drag on 1d timeframe.
Works in bull/bear via regime-adaptive logic: trend follow when chop low, avoid when chop high.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for regime context)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === KAMA (Adaptive Moving Average) on 1d close ===
    close = prices['close'].values
    # Efficiency Ratio: |close - close[10]| / sum(|diff|) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))  # length n-10
    volatility = np.abs(np.subtract(close[1:], close[:-1]))  # length n-1
    # Pad volatility for rolling sum
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    er_numerator = np.concatenate([np.full(10, np.nan), change])
    er_denominator = pd.Series(volatility_padded).rolling(window=10, min_periods=10).sum().values
    er = np.divide(er_numerator, er_denominator, out=np.full_like(er_numerator, np.nan), where=er_denominator!=0)
    # Smoothing constants: fast=2/(2+1), slow=2/(30+1)
    fast_sc = 2.0 / (2 + 1)
    slow_sc = 2.0 / (30 + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === Choppiness Index (14-period) for regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    
    # === ATR (20-period) for stoploss ===
    atr_val = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(atr_val[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Regime filter: only trend follow when chop < 61.8 (trending market)
            # Avoid signals when chop >= 61.8 (ranging/choppy market)
            trending_regime = chop[i] < 61.8
            
            # Long conditions: price > KAMA and trending regime
            long_condition = price > kama[i] and trending_regime
            # Short conditions: price < KAMA and trending regime
            short_condition = price < kama[i] and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr_val[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price closes back below KAMA (trend reversal)
            elif price < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr_val[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price closes back above KAMA (trend reversal)
            elif price > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Trend_ATRStop_v3"
timeframe = "1d"
leverage = 1.0