#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Direction_RSI_Filter_Chop_Regime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA direction: Kaufman Adaptive Moving Average
    # Parameters: ER decay = 10, SMA length = 10
    price_change = np.abs(np.diff(close, prepend=close[0]))
    er_num = np.abs(close - np.roll(close, 10))
    er_den = np.sum(price_change.reshape(-1, 1) * np.tril(np.ones((10, 10))), axis=1)
    er = np.where(er_den != 0, er_num / er_den, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = (kama > np.roll(kama, 1)).astype(float)
    
    # RSI(14) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index regime filter (14-period)
    atr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up + RSI < 50 (not overbought) + chop > 61.8 (ranging) + volume confirmation
            long_cond = (kama_dir[i] > 0.5 and 
                        rsi[i] < 50 and 
                        chop[i] > 61.8 and 
                        vol_confirm[i])
            
            # Short: KAMA down + RSI > 50 (not oversold) + chop > 61.8 (ranging) + volume confirmation
            short_cond = (kama_dir[i] < 0.5 and 
                         rsi[i] > 50 and 
                         chop[i] > 61.8 and 
                         vol_confirm[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA reverses down OR RSI > 70 (overbought)
            if kama_dir[i] < 0.5 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA reverses up OR RSI < 30 (oversold)
            if kama_dir[i] > 0.5 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily KAMA direction filter with RSI and Choppiness regime filter.
# KAMA adapts to market conditions - fast in trends, slow in ranges.
# In ranging markets (CHOP > 61.8): fade extremes (buy when RSI<50 & KAMA up, sell when RSI>50 & KAMA down).
# Volume confirmation reduces false signals.
# Works in bull markets (trend following via KAMA) and bear markets (mean reversion in ranges).
# Discrete sizing (0.25) minimizes churn. Target: 15-25 trades/year.