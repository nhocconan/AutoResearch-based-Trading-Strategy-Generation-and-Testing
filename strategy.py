#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, making it effective in both trending and ranging conditions.
On the daily timeframe, KAMA acts as a dynamic trend filter. Price above KAMA indicates an uptrend, below indicates a downtrend.
Combined with weekly trend confirmation (price above/below weekly EMA34) and volume surge (current volume > 2x 20-day average),
this strategy aims to capture strong momentum moves while avoiding whipsaws in low-volume or choppy markets.
Position size is 0.25 to limit risk and trade frequency (~10-20 trades/year).
Works in bull markets via uptrend continuation and in bear markets via downtrend continuation.
"""

name = "1d_KAMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA on daily close
    # Efficiency Ratio (ER) = |Change| / Volatility
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    # Correct volatility calculation: sum of absolute changes over ER period
    er_period = 10
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(close)
    for i in range(er_period, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (no alignment needed as already daily)
    kama_aligned = kama  # already on daily
    
    # Weekly trend filter: EMA(34) on weekly close
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if position == 0:
            # LONG: Price above KAMA and above weekly EMA34 with volume confirmation
            if (close[i] > kama_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA and below weekly EMA34 with volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or weekly EMA34
            if (close[i] < kama_aligned[i]) or \
               (close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or weekly EMA34
            if (close[i] > kama_aligned[i]) or \
               (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals