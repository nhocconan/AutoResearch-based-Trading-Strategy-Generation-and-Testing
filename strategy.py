#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 12h ADX regime filter
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Enter Long when Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (strong trend)
# - Enter Short when Bear Power > 0 AND Bull Power < 0 AND 12h ADX > 25 (strong trend)
# - Exit when power signals weaken or ADX < 20 (trend weakening)
# - Uses 12h ADX for regime filter to avoid whipsaw in ranging markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Elder Ray measures bull/bear strength relative to EMA; ADX filters for trending regimes
# - Works in both bull/bear markets by only trading strong trends

name = "6h_12h_elder_ray_power_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute Elder Ray Power from 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema13
    # Bear Power = EMA(13) - Low
    bear_power = ema13 - low
    
    # Pre-compute 12h ADX for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF ADX to LTF
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND strong uptrend (ADX > 25)
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND Bull Power < 0 AND strong downtrend (ADX > 25)
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            exit_signal = False
            # Exit long when Bull Power <= 0 OR Bear Power >= 0 OR trend weakens (ADX < 20)
            if position == 1:
                if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            # Exit short when Bear Power <= 0 OR Bull Power >= 0 OR trend weakens (ADX < 20)
            elif position == -1:
                if (bear_power[i] <= 0 or bull_power[i] >= 0 or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals