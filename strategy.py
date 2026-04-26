#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter
Hypothesis: On daily timeframe, enter long when KAMA turns up (bullish) AND RSI < 70 (avoid overbought) AND Choppiness Index > 61.8 (range regime). Enter short when KAMA turns down (bearish) AND RSI > 30 (avoid oversold) AND Choppiness Index > 61.8. Uses KAMA for adaptive trend, RSI for exhaustion filters, and Chop to avoid strong trends where mean reversion fails. Designed for low trade frequency (7-25/year) with edge in both bull and bear markets via regime-adaptive mean reversion.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter (only trade in direction of weekly trend)
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # KAMA calculation (ER=10, fastest=2, slowest=30)
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10)).values  # 10-period net change
    volatility = abs(close_series - close_series.shift(1)).rolling(window=10, min_periods=10).sum().values  # 10-period volatility
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.maximum(hh - ll, 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), Chop (14), EMA34_1w (34)
    start_idx = max(10, 14, 14, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA direction: compare current vs previous
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI filters: avoid extremes
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Chop filter: only trade in range regime (Chop > 61.8)
        chop_range = chop[i] > 61.8
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA turning up + RSI not overbought + Chop range + weekly uptrend
            long_signal = kama_up and rsi_not_overbought and chop_range and weekly_uptrend
            
            # Short: KAMA turning down + RSI not oversold + Chop range + weekly downtrend
            short_signal = kama_down and rsi_not_oversold and chop_range and weekly_downtrend
            
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
            # Exit: KAMA turns down OR RSI > 70 (overbought) OR Chop < 38.2 (trend) OR weekly trend changes
            if (not kama_up) or rsi[i] > 70 or chop[i] < 38.2 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA turns up OR RSI < 30 (oversold) OR Chop < 38.2 (trend) OR weekly trend changes
            if (not kama_down) or rsi[i] < 30 or chop[i] < 38.2 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0