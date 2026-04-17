#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Momentum_WeeklyTrendFilter
Strategy: KAMA trend direction on daily timeframe with weekly trend filter and RSI momentum.
Long: KAMA rising + weekly EMA34 > EMA144 + RSI(14) > 50
Short: KAMA falling + weekly EMA34 < EMA144 + RSI(14) < 50
Exit: KAMA direction change or RSI crosses 50
Position size: 0.25
Designed to capture medium-term trends with momentum confirmation and weekly trend filter.
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    close_series = pd.Series(close)
    # Efficiency ratio: |price change| / sum of absolute price changes
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = close_series.copy()
    for i in range(1, len(kama)):
        if not np.isnan(sc.iloc[i]):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_series.iloc[i] - kama.iloc[i-1])
        else:
            kama.iloc[i] = kama.iloc[i-1]
    kama_values = kama.values
    
    # Calculate weekly EMA34 and EMA144 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema144_1w = close_series_1w.ewm(span=144, adjust=False, min_periods=144).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema144_1w_aligned = align_htf_to_ltf(prices, df_1w, ema144_1w)
    
    # RSI(14) for momentum
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(144, 10, 14)  # max of weekly EMA144, KAMA ER period, RSI period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(ema144_1w_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(kama_values[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama_values[i] > kama_values[i-1]
        kama_falling = kama_values[i] < kama_values[i-1]
        
        # Weekly trend filter: EMA34 > EMA144 for uptrend, < for downtrend
        weekly_uptrend = ema34_1w_aligned[i] > ema144_1w_aligned[i]
        weekly_downtrend = ema34_1w_aligned[i] < ema144_1w_aligned[i]
        
        # RSI momentum: > 50 for bullish momentum, < 50 for bearish momentum
        rsi_bullish = rsi_values[i] > 50
        rsi_bearish = rsi_values[i] < 50
        
        # Entry conditions
        if position == 0:
            # Long: KAMA rising + weekly uptrend + bullish RSI momentum
            if kama_rising and weekly_uptrend and rsi_bullish:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + weekly downtrend + bearish RSI momentum
            elif kama_falling and weekly_downtrend and rsi_bearish:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling or RSI turns bearish
            if kama_falling or not rsi_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising or RSI turns bullish
            if kama_rising or rsi_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_Momentum_WeeklyTrendFilter"
timeframe = "1d"
leverage = 1.0