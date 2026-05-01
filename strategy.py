#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d regime filter.
# Bull Power = high - EMA13(close); Bear Power = EMA13(close) - low.
# Long when Bull Power > 0 AND 1d close > 1d EMA50 (bull regime).
# Short when Bear Power > 0 AND 1d close < 1d EMA50 (bear regime).
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull markets (trend following via Bull Power) and bear markets (trend following via Bear Power).
# Primary timeframe: 6h, HTF: 1d for regime filter.

name = "6h_ElderRay_BullBearPower_1dEMA50_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = high - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - low
    bear_power = ema13 - low
    
    # Load 1d data ONCE before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d close aligned for regime bias
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA13 and indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(close_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d close vs its EMA50
        bull_regime = close_1d_aligned[i] > ema50_1d_aligned[i]  # 1d close above EMA50 = bull regime
        bear_regime = close_1d_aligned[i] < ema50_1d_aligned[i]  # 1d close below EMA50 = bear regime
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power positive AND bull regime
            if bull_power[i] > 0 and bull_regime:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive AND bear regime
            elif bear_power[i] > 0 and bear_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power turns negative OR regime turns bearish
            if bull_power[i] <= 0 or bear_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power turns negative OR regime turns bullish
            if bear_power[i] <= 0 or bull_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals