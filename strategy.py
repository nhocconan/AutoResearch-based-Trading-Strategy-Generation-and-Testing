#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w EMA50 filter and volume confirmation.
# Uses KAMA (Kaufman Adaptive Moving Average) for trend detection, 1w EMA50 for higher timeframe filter,
# and volume spike for confirmation. Designed to capture strong trends while avoiding whipsaws in chop.
# Target: 15-25 trades/year to minimize fee drag and improve robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily KAMA (using close prices)
    close_1d = df_1d['close'].values
    # Efficiency ratio: |change| / sum of absolute changes
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # ER=1
    slow_sc = 2 / (30 + 1)  # ER=0
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate weekly EMA50
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily KAMA and weekly EMA50 to 1d
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need weekly EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to balance signals)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filters: price relative to KAMA and weekly EMA50
        price_above_kama = close[i] > kama_1d_aligned[i]
        price_below_kama = close[i] < kama_1d_aligned[i]
        price_above_ema = close[i] > ema50_1w_aligned[i]
        price_below_ema = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: Price above both KAMA and weekly EMA50 with volume confirmation
            if (price_above_kama and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below both KAMA and weekly EMA50 with volume confirmation
            elif (price_below_kama and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below KAMA OR below weekly EMA50
            if (close[i] < kama_1d_aligned[i]) or (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above KAMA OR above weekly EMA50
            if (close[i] > kama_1d_aligned[i]) or (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_1w_EMA50_Volume"
timeframe = "1d"
leverage = 1.0