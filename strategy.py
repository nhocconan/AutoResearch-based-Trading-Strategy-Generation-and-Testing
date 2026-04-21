#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v4
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI(2) for mean-reversion entry within trend, and Choppiness Index regime filter to avoid whipsaws. Designed for low trade frequency (~10-20/year) to minimize fee drag and improve generalization across bull/bear markets. Uses 1d primary timeframe with 1w HTF for trend confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w trend filter: 34-period EMA ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === KAMA on primary (1d) timeframe ===
    close = prices['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else 0
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(2) for mean-reversion entry ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=2, min_periods=2).mean().values
    avg_loss = pd.Series(loss).rolling(window=2, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index regime filter (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high - lowest_low) != 0,
                    100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14),
                    50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        kama_val = kama[i]
        trend_1w = ema_34_1w_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        if position == 0:
            # Long: price > KAMA (uptrend) + RSI < 20 (oversold) + chop > 61.8 (ranging market)
            if price_close > kama_val and price_close > trend_1w and rsi_val < 20 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) + RSI > 80 (overbought) + chop > 61.8 (ranging market)
            elif price_close < kama_val and price_close < trend_1w and rsi_val > 80 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse signal or chop < 38.2 (trending market - exit mean reversion)
            if position == 1 and (price_close < kama_val or chop_val < 38.2):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > kama_val or chop_val < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v4"
timeframe = "1d"
leverage = 1.0