#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_Filter
Hypothesis: Uses 1-week KAMA (Kaufman Adaptive Moving Average) to determine trend direction.
Long when price is above weekly KAMA, short when below. Entry timing uses daily RSI(14) pullbacks
to the trend (RSI < 40 in uptrend, RSI > 60 in downtrend) with volume confirmation.
Designed for low trade frequency (~10-25/year) by using weekly trend filter and
daily entry signals. Works in both bull and bear markets by following higher-timeframe trend.
"""

name = "1d_1w_KAMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s.diff(er_length))
    volatility = close_s.diff().abs().rolling(window=er_length, min_periods=er_length).sum()
    er = change / volatility
    er = er.fillna(0)
    # Smoothing Constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly KAMA for Trend Filter ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    kama_1w = calculate_kama(df_1w['close'].values, er_length=10, fast_sc=2, slow_sc=30)
    kama_1w_1d = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # --- Daily RSI for Entry Timing ---
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # --- Volume Confirmation ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_1d[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly KAMA
        above_kama = close[i] > kama_1w_1d[i]
        below_kama = close[i] < kama_1w_1d[i]
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price above weekly KAMA + RSI pullback (<40) + volume
            if (above_kama and 
                rsi[i] < 40 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly KAMA + RSI pullback (>60) + volume
            elif (below_kama and 
                  rsi[i] > 60 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or RSI extreme in opposite direction
            if position == 1:
                # Exit long: price below weekly KAMA OR RSI > 70 (overbought)
                if below_kama or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price above weekly KAMA OR RSI < 30 (oversold)
                if above_kama or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals