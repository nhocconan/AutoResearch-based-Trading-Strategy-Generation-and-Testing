#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI(14) for momentum confirmation, and Choppiness Index (CHOP) as regime filter to avoid whipsaws. Enter long when KAMA rising, RSI>50, and CHOP<61.8 (trending regime); short when KAMA falling, RSI<50, and CHOP<61.8. Exit on opposite signal. Uses 1d primary with 1w HTF for higher-timeframe trend alignment. Designed for low trade frequency (~10-20/year) to minimize fee drag and improve generalization across bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop: 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for higher-timeframe trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d indicators: KAMA, RSI(14), Choppiness Index(14) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- KAMA (Efficiency Ratio = 10, Fast=2, Slow=30) ---
    change = np.abs(np.diff(close, k=10))  # |close - close[10]|
    change[0:10] = np.nan  # not enough data for first 10 bars
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close - close[1]| over 10 bars
    # manual rolling sum of volatility
    volatility_sum = np.zeros_like(close)
    for i in range(10, n):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    volatility_sum[0:10] = np.nan
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)  # efficiency ratio
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- RSI(14) ---
    delta = np.diff(close)
    delta = np.insert(delta, 0, np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Choppiness Index(14) ---
    # true range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high - lowest_low) != 0,
                    -100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14),
                    50)
    
    # Align 1d indicators (they are already LTF, but we align for consistency)
    kama_aligned = kama  # no alignment needed for LTF
    rsi_aligned = rsi
    chop_aligned = chop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # warmup for 1w EMA34 and 1d indicators
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher-timeframe trend filter: price above/below 1w EMA34
        htf_uptrend = close[i] > ema_34_1w_aligned[i]
        htf_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # 1d conditions
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        chop_trending = chop_aligned[i] < 61.8  # trending regime
        
        if position == 0:
            # Long: KAMA rising, RSI>50, trending regime, and HTF uptrend
            if kama_rising and rsi_above_50 and chop_trending and htf_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI<50, trending regime, and HTF downtrend
            elif kama_falling and rsi_below_50 and chop_trending and htf_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling OR RSI<50 OR choppy regime (CHOP>61.8)
            if (not kama_rising) or (rsi_aligned[i] < 50) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising OR RSI>50 OR choppy regime (CHOP>61.8)
            if (not kama_falling) or (rsi_aligned[i] > 50) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0