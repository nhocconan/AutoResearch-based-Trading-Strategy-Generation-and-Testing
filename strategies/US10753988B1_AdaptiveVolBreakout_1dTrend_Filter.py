# US Patent US10753988B1 - Novel Adaptive Volatility Breakout with Volatility Regime Filter
# Strategy: Uses a novel volatility breakout system (patent-inspired) combined with a volatility regime filter
# Timeframe: 4h
# Volatility breakout triggers on expansion beyond adaptive Bollinger Bands (using ATR-based deviation)
# Volatility regime filter uses ATR ratio to distinguish between trending and ranging markets
# Designed to work in both bull and bear markets by filtering trades based on volatility regime
# Entry conditions: Volatility breakout + volatility regime alignment (trending market)
# Exit conditions: Volatility contraction or opposite breakout signal
# Position sizing: Discrete levels (0.25) to minimize churn
# Uses 1d trend filter for higher timeframe bias

#!/usr/bin/env python3
name = "US10753988B1_AdaptiveVolBreakout_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Adaptive Bollinger Bands using ATR for dynamic deviation
    # Calculate ATR(21)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate adaptive deviation: ATR * volatility multiplier
    vol_mult = 2.0  # Base multiplier
    dev = atr * vol_mult
    
    # Calculate middle band: SMA(21)
    close_pd = pd.Series(close)
    sma21 = close_pd.rolling(window=21, min_periods=21).mean().values
    
    # Upper and lower bands
    upper_band = sma21 + dev
    lower_band = sma21 - dev
    
    # Volatility regime filter: ATR ratio (current ATR vs longer ATR)
    atr_long = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr / atr_long  # >1 indicates expanding volatility (trending), <1 indicates contracting (ranging)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for longest indicator
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band AND volatility expanding (atr_ratio > 1.1) AND 1d uptrend
            if (close[i] > upper_band[i] and 
                atr_ratio[i] > 1.1 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band AND volatility expanding (atr_ratio > 1.1) AND 1d downtrend
            elif (close[i] < lower_band[i] and 
                  atr_ratio[i] > 1.1 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below lower band OR volatility contracting (atr_ratio < 0.9)
            if (close[i] < lower_band[i] or atr_ratio[i] < 0.9):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above upper band OR volatility contracting (atr_ratio < 0.9)
            if (close[i] > upper_band[i] or atr_ratio[i] < 0.9):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals