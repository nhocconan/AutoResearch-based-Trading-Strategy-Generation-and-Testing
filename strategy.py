#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 1d EMA34 trend and 4h Bollinger Band mean reversion
# Uses daily EMA34 for trend filter, 4h Bollinger Bands (20,2) for entry/exit,
# and Choppiness Index (14) to identify ranging markets for mean reversion.
# Works in sideways markets (reversion to mean at BB extremes) and avoids trending markets.
# Designed for 20-40 trades/year to avoid fee drag.
name = "4h_Choppiness_BollingerMeanReversion_1dEMA34"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4h Bollinger Bands (20,2)
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # 4h Choppiness Index (14)
    atr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    atr1[0] = high[0] - low[0]
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_4h[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range condition: Choppiness Index > 61.8 (ranging market)
        range_market = chop[i] > 61.8
        
        if position == 0:
            # Long: Price at lower Bollinger Band in ranging market with uptrend bias
            if range_market and close[i] <= bb_lower[i] and close[i] > ema34_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price at upper Bollinger Band in ranging market with downtrend bias
            elif range_market and close[i] >= bb_upper[i] and close[i] < ema34_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price reaches middle Bollinger Band OR trend turns down
            if close[i] >= bb_mid[i] or close[i] < ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price reaches middle Bollinger Band OR trend turns up
            if close[i] <= bb_mid[i] or close[i] > ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals