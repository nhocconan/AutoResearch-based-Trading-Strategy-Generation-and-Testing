#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Filter
Hypothesis: On daily timeframe, KAMA identifies adaptive trend direction while RSI(14) provides mean-reversion signals within the trend. Trades are taken only when KAMA trend and RSI extreme align, filtered by weekly trend (1w EMA50) and volume confirmation. Designed for low frequency (7-25 trades/year) to minimize fee impact and work in both bull and bear markets by avoiding counter-trend trades.
"""
name = "1d_KAMA_Trend_RSI_Filter"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # Daily RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.3 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure KAMA, RSI, vol_avg are valid
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI oversold (<30) + weekly uptrend + volume
            if close[i] > kama[i] and rsi[i] < 30 and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + RSI overbought (>70) + weekly downtrend + volume
            elif close[i] < kama[i] and rsi[i] > 70 and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal
            if position == 1:
                if rsi[i] > 40 or close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi[i] < 60 or close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals