#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 12h trend filter
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND price > 12h EMA50
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 AND price < 12h EMA50
# Uses Williams Alligator (SMAs with specific periods) to identify trend alignment and Elder Ray to measure bull/bear power relative to EMA13.
# 12h EMA50 acts as higher-timeframe trend filter to avoid counter-trend whipsaws.
# Discrete position sizing (0.25) to control fee drag and drawdown.
# Target: 12-30 trades/year (~50-120 total over 4 years) to minimize fee impact on 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Warmup calculation for Alligator (13,8,5 SMAs) and EMA50
    warmup = 50  # sufficient for all indicators
    
    # === Williams Alligator (6h) ===
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA (Smoothed Moving Average) approximation using EMA with alpha=1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # === Elder Ray (6h) ===
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 12h HTF: EMA50 trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish alignment: Jaw < Teeth < Lips
        # 2. Bull Power > 0 (bulls in control)
        # 3. Price > 12h EMA50 (higher-timeframe uptrend)
        if (jaw[i] < teeth[i]) and (teeth[i] < lips[i]) and \
           (bull_power[i] > 0) and (close[i] > ema_50_12h_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish alignment: Jaw > Teeth > Lips
        # 2. Bear Power < 0 (bears in control)
        # 3. Price < 12h EMA50 (higher-timeframe downtrend)
        elif (jaw[i] > teeth[i]) and (teeth[i] > lips[i]) and \
             (bear_power[i] < 0) and (close[i] < ema_50_12h_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Alligator_ElderRay_12hEMA50_v1"
timeframe = "6h"
leverage = 1.0