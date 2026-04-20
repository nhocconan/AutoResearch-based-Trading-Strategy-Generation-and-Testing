#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d Volatility Regime Filter
# - Williams %R(14) on 6h for overbought/oversold signals
# - 1d ATR ratio (ATR10/ATR30) as volatility regime filter
# - Only take longs when ATR ratio > 1.3 (high volatility) and W%R < -80
# - Only take shorts when ATR ratio > 1.3 and W%R > -20
# - High volatility environments improve mean reversion edge in crypto
# - Volatility filter avoids choppy low-vol periods where mean reversion fails
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR components for volatility regime
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # first period
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    # Avoid division by zero
    atr_ratio = np.where(atr30 != 0, atr10 / atr30, 1.0)
    atr_ratio_6h = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate Williams %R on 6h
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    wr = np.where(
        (highest_high - lowest_low) != 0,
        -100 * (highest_high - close_6h) / (highest_high - lowest_low),
        -50  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if NaN in indicators
        if np.isnan(wr[i]) or np.isnan(atr_ratio_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime: only trade in high volatility (ATR ratio > 1.3)
        high_vol = atr_ratio_6h[i] > 1.3
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + high volatility
            if wr[i] < -80 and high_vol:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + high volatility
            elif wr[i] > -20 and high_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or volatility drops
            if wr[i] > -50 or not high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or volatility drops
            if wr[i] < -50 or not high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_VolatilityRegime"
timeframe = "6h"
leverage = 1.0