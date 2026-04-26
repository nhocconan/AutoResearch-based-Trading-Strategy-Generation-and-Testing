#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop_Regime
Hypothesis: On 1d timeframe, enter long when KAMA(14,2,30) turns up AND RSI(14) > 50 AND Choppiness Index(14) < 38.2 (trending regime). Enter short when KAMA turns down AND RSI(14) < 50 AND Choppiness Index(14) < 38.2. Uses KAMA for adaptive trend, RSI for momentum confirmation, and Choppiness Index to avoid ranging markets. Designed for low trade frequency (10-25/year) to minimize fee drag while capturing sustained trends in both bull and bear markets. Uses 1w EMA50 as higher timeframe trend filter to avoid counter-trend trades.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # KAMA: Adaptive Moving Average
    # Efficiency Ratio (ER) = |Change| / Sum(|Changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants: fastest SC=2/(2+1)=0.6667, slowest SC=2/(30+1)=0.0645
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14): measures whether market is choppy (sideways) or trending
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(N)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.maximum(max_high - min_low, 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), CHOP (14), EMA50 (50)
    start_idx = max(10, 14, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA direction: comparing current vs previous
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # Trend regime: CHOP < 38.2 indicates trending market
        trending_regime = chop[i] < 38.2
        
        if position == 0:
            # Long: KAMA up AND RSI > 50 AND trending regime AND 1w uptrend
            long_signal = kama_up and (rsi[i] > 50) and trending_regime and (close[i] > ema_50_1w_aligned[i])
            
            # Short: KAMA down AND RSI < 50 AND trending regime AND 1w downtrend
            short_signal = kama_down and (rsi[i] < 50) and trending_regime and (close[i] < ema_50_1w_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA down OR choppy regime OR trend change to downtrend
            if kama_down or (chop[i] >= 38.2) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA up OR choppy regime OR trend change to uptrend
            if kama_up or (chop[i] >= 38.2) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0