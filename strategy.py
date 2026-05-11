#!/usr/bin/env python3
"""
4h_Choppiness_RSI_MeanReversion_v1
Hypothesis: Uses RSI mean-reversion signals filtered by Choppiness Index regime.
In choppy markets (Choppiness > 61.8), we take long positions when RSI < 30 and short when RSI > 70.
In trending markets (Choppiness < 38.2), we avoid trades to prevent whipsaw.
Designed for low trade frequency (~20-30 trades/year) by requiring high Choppiness and extreme RSI.
Works in both bull and bear markets by focusing on mean reversion in ranging conditions.
"""

name = "4h_Choppiness_RSI_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- RSI (14-period) on 4h close ---
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # --- Choppiness Index (14-period) on 1d data ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(tr_sum / (atr * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)  # Replace NaN with neutral value
    
    # Align Choppiness Index to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(chop_aligned[i]):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filters
        choppy_market = chop_aligned[i] > 61.8  # Ranging market
        trending_market = chop_aligned[i] < 38.2  # Trending market
        
        if position == 0:
            # Only trade in choppy/ranging markets
            if choppy_market:
                # RSI mean reversion signals
                if rsi[i] < 30:  # Oversold -> long
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:  # Overbought -> short
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions: RSI returns to neutral zone or regime changes
            if position == 1:
                # Exit long: RSI > 50 or market becomes trending
                exit_signal = (rsi[i] > 50) or trending_market
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI < 50 or market becomes trending
                exit_signal = (rsi[i] < 50) or trending_market
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals