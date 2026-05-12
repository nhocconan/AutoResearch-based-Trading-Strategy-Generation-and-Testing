#!/usr/bin/env python3
# 1D_CHOPPINESS_INDEX_MEAN_REVERSION
# Hypothesis: Use weekly Choppiness Index to detect ranging markets (Chop > 61.8) and apply mean reversion on daily timeframe.
# In ranging markets, price tends to revert to the mean (VWAP). Enter long when price < VWAP - 0.5*ATR, short when price > VWAP + 0.5*ATR.
# Exit when price crosses VWAP. This avoids trending markets where mean reversion fails.
# Works in both bull and bear markets by focusing on ranging regimes.
# Target: 15-25 trades/year on 1d timeframe.

name = "1D_CHOPPINESS_INDEX_MEAN_REVERSION"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP
    typical_price = (high + low + close) / 3
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    
    # Calculate ATR(14)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Weekly data for Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range for weekly
    tr1_w = np.abs(high_1w - low_1w)
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1_w[0] = 0
    tr2_w[0] = 0
    tr3_w[0] = 0
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    
    # Choppiness Index: 100 * log10(sum(TR, 14) / (ATR(14) * 14)) / log10(14)
    atr_w = pd.Series(tr_w).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_tr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_w / (atr_w * 14)) / np.log10(14)
    
    # Align weekly Chop to daily
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Need enough data for ATR and Chop
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (Chop > 61.8)
        if chop_aligned[i] > 61.8:
            if position == 0:
                # LONG: Price below VWAP - 0.5*ATR
                if close[i] < vwap[i] - 0.5 * atr[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Price above VWAP + 0.5*ATR
                elif close[i] > vwap[i] + 0.5 * atr[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Price crosses above VWAP
                if close[i] > vwap[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: Price crosses below VWAP
                if close[i] < vwap[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In trending markets, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals