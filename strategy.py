#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA200 trend filter and ATR-based volatility filter.
Long when Bull Power > 0 AND price > 1d EMA200 AND ATR(14) < 1.5 * ATR(50) (low volatility regime).
Short when Bear Power < 0 AND price < 1d EMA200 AND ATR(14) < 1.5 * ATR(50).
Exit when Elder Power reverses sign OR volatility expands (ATR(14) > 2.0 * ATR(50)).
Elder Ray identifies bull/bear strength relative to EMA13; combined with trend and volatility filters,
it avoids whipsaws in ranging markets and captures sustained moves in both bull and bear regimes.
Uses proven Elder Ray concept from institutional trading adapted to crypto with proper risk control.
Target: 50-150 total trades over 4 years = 12-37/year.
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
    
    # Get 1d data for EMA200 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d timeframe
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate EMA13 for Elder Ray (on 6h)
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ATR(14) and ATR(50) for volatility regime filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    tr_series = pd.Series(tr)
    atr_14 = tr_series.rolling(window=14, min_periods=14).mean().values
    atr_50 = tr_series.rolling(window=50, min_periods=50).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Align all indicators to 6h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)  # EMA13 needs same alignment as EMA200
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(ema_13_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_200 = ema_200_1d_aligned[i]
        ema_13_val = ema_13_aligned[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        atr14 = atr_14_aligned[i]
        atr50 = atr_50_aligned[i]
        price = close[i]
        
        # Volatility regime: low volatility = ATR14 < 1.5 * ATR50
        low_vol = atr14 < 1.5 * atr50
        # High volatility exit = ATR14 > 2.0 * ATR50
        high_vol_exit = atr14 > 2.0 * atr50
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) AND price > 1d EMA200 (uptrend) AND low volatility
            if bull > 0 and price > ema_200 and low_vol:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) AND price < 1d EMA200 (downtrend) AND low volatility
            elif bear < 0 and price < ema_200 and low_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power < 0 OR high volatility expansion
            if bear < 0 or high_vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power > 0 OR high volatility expansion
            if bull > 0 or high_vol_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA200_VolatilityFilter"
timeframe = "6h"
leverage = 1.0