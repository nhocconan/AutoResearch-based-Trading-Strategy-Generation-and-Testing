#!/usr/bin/env python3
# 1D_Keltner_RSI_Breakout
# Hypothesis: On daily timeframe, enter long when price breaks above Keltner upper band (EMA20 + 2*ATR) with RSI > 50, short when breaks below lower band (EMA20 - 2*ATR) with RSI < 50. Use weekly EMA50 as trend filter to avoid counter-trend trades. Target: 10-25 trades/year per symbol (40-100 total over 4 years) with low frequency to minimize fee drag. Works in bull (breakouts) and bear (mean reversion at extremes) markets.

name = "1D_Keltner_RSI_Breakout"
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
    
    # Calculate EMA20 and ATR(14) for Keltner channels
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range and ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Keltner channels
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1w = close_1w > ema50_1w
    
    # Align weekly trend to daily
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or np.isnan(rsi[i]) or np.isnan(trend_up_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Keltner upper + RSI > 50 + weekly uptrend
            if close[i] > keltner_upper[i] and rsi[i] > 50 and trend_up_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Keltner lower + RSI < 50 + weekly downtrend
            elif close[i] < keltner_lower[i] and rsi[i] < 50 and not trend_up_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Keltner lower (reversal) or RSI < 40
            if close[i] < keltner_lower[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Keltner upper (reversal) or RSI > 60
            if close[i] > keltner_upper[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals