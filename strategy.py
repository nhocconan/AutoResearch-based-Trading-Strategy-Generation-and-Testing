#!/usr/bin/env python3
"""
4h_RSI_Stochastic_BullBear_Trap
Hypothesis: In strong trends, price often pulls back to key momentum zones (RSI 40-60) before continuing. Combine with Stochastic to identify overextended pullbacks. Enter long when RSI>50 and Stochastic crosses up from oversold in uptrend; short when RSI<50 and Stochastic crosses down from overbought in downtrend. Use 1-day ADX trend filter to avoid chop. Works in bull/bear via trend filter. Target: 20-30 trades/year.
"""

name = "4h_RSI_Stochastic_BullBear_Trap"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - high_1d[:-1]), 
                               np.abs(low_1d[1:] - low_1d[:-1])))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / np.concatenate([[np.nan], atr_1d[:-1]])
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / np.concatenate([[np.nan], atr_1d[:-1]])
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 4h data for RSI and Stochastic
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic(14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 14 periods for RSI/Stoch, plus 1 for DM calculation
    start_idx = 15
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(k_percent[i]) or
            np.isnan(d_percent[i]) or
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            if trending:
                # Long: RSI > 50 (bullish momentum) and Stochastic crosses up from oversold
                if rsi[i] > 50 and k_percent[i-1] <= 20 and k_percent[i] > 20 and k_percent[i] > d_percent[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: RSI < 50 (bearish momentum) and Stochastic crosses down from overbought
                elif rsi[i] < 50 and k_percent[i-1] >= 80 and k_percent[i] < 80 and k_percent[i] < d_percent[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: RSI < 40 (loss of momentum) or Stochastic overbought
            if rsi[i] < 40 or (k_percent[i] > 80 and d_percent[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI > 60 (loss of bearish momentum) or Stochastic oversold
            if rsi[i] > 60 or (k_percent[i] < 20 and d_percent[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals