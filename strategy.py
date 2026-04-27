#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_MeanReversion_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for mean-reversion entries, and Choppiness Index to filter regimes.
Long when KAMA trending up, RSI < 30 (oversold), and CHOP > 61.8 (ranging market).
Short when KAMA trending down, RSI > 70 (overbought), and CHOP > 61.8 (ranging market).
This strategy profits from mean reversion in ranging markets while avoiding trend-following whipsaws.
Designed for low trade frequency (~10-20 trades/year) on 1d to minimize fee drag and work in both bull and bear markets.
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
    volume = prices['volume'].values
    
    # Calculate 1w data ONCE before loop for trend filter (optional, can use 1d close vs EMA)
    # We'll use 1d EMA50 as trend filter instead of 1w for simplicity and stability
    
    # Calculate KAMA on 1d close
    # KAMA parameters: ER = 10, fast = 2, slow = 30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Correct calculation of efficiency ratio
    dir = np.abs(np.diff(close, n=10, prepend=close[:10]))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # recalc below
    
    # Recalculate volatility properly: sum of absolute daily changes over 10 periods
    daily_changes = np.abs(np.diff(close, prepend=close[0]))
    vol = np.zeros_like(close)
    for i in range(10, len(close)):
        vol[i] = np.sum(daily_changes[i-9:i+1])  # 10-period sum
    
    # Avoid division by zero
    er = np.where(vol > 0, dir / vol, 0)
    # Smoothing constants
    fast = 2.0
    slow = 30.0
    sc = (er * (fast/slow - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    hl_range = hh - ll
    
    # Avoid division by zero and log(0)
    chop = np.where((hl_range > 0) & (tr_sum > 0), 
                    100 * np.log10(tr_sum / hl_range) / np.log10(14), 
                    50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for KAMA (10), RSI (14), CHOP (14)
    start_idx = max(10, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        if position == 0:
            # Flat - look for entry: mean reversion in ranging market
            # Long: Price below KAMA (dip), RSI < 30 (oversold), CHOP > 61.8 (ranging)
            # Short: Price above KAMA (rally), RSI > 70 (overbought), CHOP > 61.8 (ranging)
            long_condition = (close_val < kama_val and 
                            rsi_val < 30 and 
                            chop_val > 61.8)
            short_condition = (close_val > kama_val and 
                             rsi_val > 70 and 
                             chop_val > 61.8)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long - exit when price crosses above KAMA (mean reversion complete) OR trend changes
            if close_val > kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price crosses below KAMA (mean reversion complete) OR trend changes
            if close_val < kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_RSI_MeanReversion_ChopFilter"
timeframe = "1d"
leverage = 1.0