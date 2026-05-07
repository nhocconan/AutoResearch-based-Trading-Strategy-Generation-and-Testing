#!/usr/bin/env python3
# 6h_Illuminated_Signal_60
# Hypothesis: 6h chart strategy using a composite signal from three independent but complementary indicators:
#   1) 6h RSI(2) < 10 for mean reversion long signals (oversold)
#   2) 6h RSI(2) > 90 for mean reversion short signals (overbought)
#   3) 1d Williams %R > -20 for bullish bias (market strength)
#   4) 1d Williams %R < -80 for bearish bias (market weakness)
# Entry requires both the short-term RSI extreme and the 1d Williams %R bias aligned.
# Exit when RSI(2) crosses back to neutral (50) or opposite extreme.
# Williams %R acts as a regime filter to avoid counter-trend trades in strong moves.
# This combination aims to capture mean reversion within the prevailing daily trend,
# working in both bull (buy dips) and bear (sell rallies) markets.
# Target: ~20-40 trades/year to minimize fee drag while maintaining edge.

timeframe = "6h"
name = "6h_Illuminated_Signal_60"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate RSI(2) on 6h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing: alpha = 1/period
    alpha = 1.0 / 2
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Get daily data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, ((highest_high - close_1d) / hh_ll) * -100, -50)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(2, 14)  # Need RSI(2) and Williams %R(14) data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi[i]) or np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) AND Williams %R > -20 (not overbought on daily)
            if (rsi[i] < 10 and williams_r_aligned[i] > -20):
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 (overbought) AND Williams %R < -80 (not oversold on daily)
            elif (rsi[i] > 90 and williams_r_aligned[i] < -80):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI(2) crosses back above 50 (mean reversion complete) or becomes overbought
            if rsi[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI(2) crosses back below 50 (mean reversion complete) or becomes oversold
            if rsi[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals