#!/usr/bin/env python3
"""
Experiment #7951: 6-hour price action with 1-day volatility regime filter.
Hypothesis: In low volatility regimes (1-day ATR ratio < 0.8), price tends to mean revert at 
6-hour support/resistance levels. In high volatility regimes (ATR ratio >= 0.8), price breaks 
out of 6-hour ranges. This adapts to both trending and ranging markets. 
Target: 100-200 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7951_6h_vol_regime_meanrev_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
SUPPORT_RESISTANCE_PERIOD = 20
VOLATILITY_LOOKBACK = 30
VOLATILITY_SHORT = 7
VOLATILITY_THRESHOLD = 0.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(np.roll(high_1d, 1) - close_1d)
    tr3_1d = np.abs(np.roll(low_1d, 1) - close_1d)
    tr1_1d = pd.Series(tr1_1d)
    tr2_1d = pd.Series(tr2_1d)
    tr3_1d = pd.Series(tr3_1d)
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d_short = tr_1d.ewm(span=VOLATILITY_SHORT, adjust=False, min_periods=VOLATILITY_SHORT).mean().values
    atr_1d_long = tr_1d.ewm(span=VOLATILITY_LOOKBACK, adjust=False, min_periods=VOLATILITY_LOOKBACK).mean().values
    
    # Volatility ratio: short-term ATR / long-term ATR
    vol_ratio = np.where(atr_1d_long > 0, atr_1d_short / atr_1d_long, 1.0)
    high_vol_regime = vol_ratio >= VOLATILITY_THRESHOLD  # True = breakout mode
    low_vol_regime = vol_ratio < VOLATILITY_THRESHOLD    # True = mean reversion mode
    vol_regime_high_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    vol_regime_low_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Support/resistance levels (6-hour high/low)
    resistance = pd.Series(high).rolling(window=SUPPORT_RESISTANCE_PERIOD, min_periods=SUPPORT_RESISTANCE_PERIOD).max().values
    support = pd.Series(low).rolling(window=SUPPORT_RESISTANCE_PERIOD, min_periods=SUPPORT_RESISTANCE_PERIOD).min().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SUPPORT_RESISTANCE_PERIOD, VOLATILITY_LOOKBACK, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(vol_regime_high_aligned[i]) or np.isnan(vol_regime_low_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        high_vol = vol_regime_high_aligned[i]
        low_vol = vol_regime_low_aligned[i]
        
        # Price near support/resistance (within 0.5*ATR)
        near_support = (low[i] <= support[i-1] + 0.5 * atr[i-1]) and (i-1 >= 0) and not np.isnan(support[i-1])
        near_resistance = (high[i] >= resistance[i-1] - 0.5 * atr[i-1]) and (i-1 >= 0) and not np.isnan(resistance[i-1])
        
        # Breakout conditions
        upside_breakout = (close[i] > resistance[i-1]) and (i-1 >= 0) and not np.isnan(resistance[i-1])
        downside_breakout = (close[i] < support[i-1]) and (i-1 >= 0) and not np.isnan(support[i-1])
        
        # Entry logic based on volatility regime
        if position == 0:
            if low_vol and near_support:
                # Mean reversion long from support in low vol
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif low_vol and near_resistance:
                # Mean reversion short from resistance in low vol
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif high_vol and upside_breakout:
                # Breakout long in high vol
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif high_vol and downside_breakout:
                # Breakout short in high vol
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals