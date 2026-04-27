#!/usr/bin/env python3
"""
1d_Keltner_RSI_Trend_Filter
Hypothesis: Price touching Keltner Channel bands with RSI trend filter captures reversals in both bull and bear markets. Uses weekly trend alignment and volume confirmation to avoid whipsaws. Designed for low trade frequency (~10-20 trades/year) to minimize fee drag on daily timeframe.
"""

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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly trend: EMA20
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Keltner Channel (20, 2.0) on daily
    atr_period = 20
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    ema20_close = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema20_close + 2.0 * atr
    kc_lower = ema20_close - 2.0 * atr
    
    # RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(rsi[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        weekly_trend_up = close[i] > ema20_1w_aligned[i]
        weekly_trend_down = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # Long: price touches lower KC, RSI oversold, weekly uptrend, volume confirmation
            if (close[i] <= kc_lower[i] and rsi[i] < 30 and 
                weekly_trend_up and vol_confirm[i]):
                signals[i] = size
                position = 1
            # Short: price touches upper KC, RSI overbought, weekly downtrend, volume confirmation
            elif (close[i] >= kc_upper[i] and rsi[i] > 70 and 
                  weekly_trend_down and vol_confirm[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches upper KC or RSI overbought
            if close[i] >= kc_upper[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches lower KC or RSI oversold
            if close[i] <= kc_lower[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Keltner_RSI_Trend_Filter"
timeframe = "1d"
leverage = 1.0