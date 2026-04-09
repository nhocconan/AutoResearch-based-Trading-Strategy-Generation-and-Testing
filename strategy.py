#!/usr/bin/env python3
# 1d_kama_rsi_chop_regime_v1
# Hypothesis: Daily strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
# RSI for momentum confirmation, and Choppiness Index for regime filtering.
# Long when KAMA slope > 0, RSI > 50, and chop < 61.8 (trending market).
# Short when KAMA slope < 0, RSI < 50, and chop < 61.8 (trending market).
# Exit when any condition fails.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 15-25 trades/year (60-100 total over 4 years) on BTC/ETH/SOL.
# Works in both bull and bear markets: KAMA adapts to volatility, RSI confirms momentum,
# chop filter avoids whipsaws in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (ER=10, fastest=2, slowest=30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))  # 10-period net change
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()  # 10-period volatility
    er = change / volatility.replace(0, 1e-10)  # Efficiency Ratio
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # Smoothing Constant
    kama = [close_s.iloc[0]]  # Initialize with first close
    for i in range(1, len(close_s)):
        kama.append(kama[-1] + sc.iloc[i] * (close_s.iloc[i] - kama[-1]))
    kama = np.array(kama)
    
    # KAMA slope (1-period difference)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # RSI (14-period)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = low_series.rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(atr_sum / np.log10(atr_period) / (highest_high - lowest_low))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_slope[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: KAMA slope <= 0 OR RSI <= 50 OR chop >= 61.8 (range)
            if kama_slope[i] <= 0 or rsi[i] <= 50 or chop[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA slope >= 0 OR RSI >= 50 OR chop >= 61.8 (range)
            if kama_slope[i] >= 0 or rsi[i] >= 50 or chop[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry conditions
            long_entry = (kama_slope[i] > 0) and (rsi[i] > 50) and trending_market
            short_entry = (kama_slope[i] < 0) and (rsi[i] < 50) and trending_market
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals