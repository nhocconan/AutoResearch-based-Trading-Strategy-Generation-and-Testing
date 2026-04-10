#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1d ADX regime filter
# - Bull Power = High - EMA13(1d), Bear Power = EMA13(1d) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (strong trend)
# - Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (strong trend)
# - Exit when power values converge (|Bull Power| < 0.1 * ATR1d AND |Bear Power| < 0.1 * ATR1d)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear strength relative to 1d EMA13; ADX filters for trending regimes only
# - Works in both bull and bear markets by only taking trades in strong trend directions

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute Elder Ray components: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Pre-compute 1d ATR for exit threshold
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align length
    atr1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr1d_aligned = align_htf_to_ltf(prices, df_1d, atr1d)
    
    # Pre-compute 1d ADX for regime filter
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_abs = np.abs(tr)
    plus_di14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / \
                pd.Series(tr_abs).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_di14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / \
                 pd.Series(tr_abs).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(atr1d_aligned[i]) or np.isnan(adx_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong uptrend)
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (strong downtrend)
            elif (bear_power_aligned[i] > 0 and 
                  bull_power_aligned[i] < 0 and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power converges
            # Exit when both powers are near zero (|power| < 0.1 * ATR1d)
            power_threshold = 0.1 * atr1d_aligned[i]
            if (np.abs(bull_power_aligned[i]) < power_threshold and 
                np.abs(bear_power_aligned[i]) < power_threshold):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals