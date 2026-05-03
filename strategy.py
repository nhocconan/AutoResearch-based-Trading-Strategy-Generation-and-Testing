#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and ATR-based regime
# Elder Ray measures bull/bear power relative to EMA13; avoids whipsaws by trading only with 1d trend
# ATR regime filter: only trade when ATR(14) < 1.5 * ATR(50) to avoid high volatility chop
# Works in bull/bear: 1d EMA50 ensures alignment with higher timeframe direction
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag

name = "6h_ElderRay_1dEMA50_ATRRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate ATR(14) and ATR(50) for regime filter (6h)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR regime: only trade when volatility is contracting (ATR14 < 1.5 * ATR50)
        atr_regime = atr14[i] < (1.5 * atr50[i])
        
        # Elder Ray signals with 1d trend filter and ATR regime
        # Long: Bull Power > 0 (strong bulls) + price above 1d EMA50 + ATR regime
        # Short: Bear Power < 0 (strong bears) + price below 1d EMA50 + ATR regime
        if position == 0:
            if (bull_power[i] > 0 and 
                close[i] > ema_50_1d_aligned[i] and atr_regime):
                signals[i] = 0.25
                position = 1
            elif (bear_power[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and atr_regime):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 (bulls weakening) OR price below 1d EMA50
            if bull_power[i] <= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 (bears weakening) OR price above 1d EMA50
            if bear_power[i] >= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals