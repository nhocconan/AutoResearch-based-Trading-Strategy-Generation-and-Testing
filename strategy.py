#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_With_RSI_and_Chop
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
Combined with RSI for momentum confirmation and Choppiness Index to avoid whipsaws in ranging markets.
Designed for low turnover (7-25 trades/year) on 1d timeframe to minimize fee drag.
"""

name = "1d_KAMA_Trend_Filter_With_RSI_and_Chop"
timeframe = "1d"
leverage = 1.0

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
    
    # === KAMA Trend (10-period ER) ===
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.sum, 'axis') else np.abs(np.diff(close)).cumsum()
    # Fix volatility calculation - it should be cumulative sum of absolute changes
    volatility = np.nancumsum(np.abs(np.diff(close, prepend=close[0])))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Avoid division by zero
    range_hl = max_high - min_low
    chop = np.where(range_hl != 0, 100 * np.log10(sum_atr / range_hl) / np.log10(14), 50)
    
    # === 1-week Trend Filter (EMA50) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for indicators)
    start_idx = 50  # covers KAMA and EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA + RSI > 50 + Chop < 61.8 (trending) + above weekly EMA50
            if (close[i] > kama[i] and rsi[i] > 50 and chop[i] < 61.8 and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = position_size
                position = 1
            # Short: Price below KAMA + RSI < 50 + Chop < 61.8 (trending) + below weekly EMA50
            elif (close[i] < kama[i] and rsi[i] < 50 and chop[i] < 61.8 and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below KAMA OR RSI < 40 OR Chop > 61.8 (ranging)
                if (close[i] < kama[i] or rsi[i] < 40 or chop[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above KAMA OR RSI > 60 OR Chop > 61.8 (ranging)
                if (close[i] > kama[i] or rsi[i] > 60 or chop[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals