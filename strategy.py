#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_MeanReversion_ChopFilter_v1
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for mean-reversion entries, and Choppiness Index(14) for regime filtering.
Long when: KAMA trending up, price pulls back to KAMA (RSI < 40), and market is choppy (CHOP > 61.8).
Short when: KAMA trending down, price bounces to KAMA (RSI > 60), and market is choppy (CHOP > 61.8).
Uses discrete position sizing (0.25) to minimize fee churn. Designed to work in ranging markets
which frequently occur in bear markets (2022, 2025) while avoiding strong trends where mean reversion fails.
Target: 20-60 trades/year (80-240 total over 4 years) with strict regime filter to avoid overtrading.
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
    
    # Get 1w data for HTF trend filter (only trade in direction of weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === KAMA (Kaufman Adaptive Moving Average) on 1d close ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 1d (already on 1d, but using align_htf_to_ltf for consistency with rule)
    df_1d = get_htf_data(prices, '1d')
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === RSI(14) on 1d close ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    avg_gain[13] = np.mean(gain[1:14])  # first average
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan, dtype=float), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 1d
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === Choppiness Index(14) on 1d OHLC ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high[0] - low[0]  # first TR
    # Sum of TR over 14 periods
    atr14 = np.zeros_like(close)
    for i in range(14, n):
        atr14[i] = np.sum(tr[i-13:i+1])
    # Highest high and lowest low over 14 periods
    hh14 = np.zeros_like(close)
    ll14 = np.zeros_like(close)
    for i in range(14, n):
        hh14[i] = np.max(high[i-13:i+1])
        ll14[i] = np.min(low[i-13:i+1])
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    range14 = hh14 - ll14
    chop = np.full_like(close, np.nan, dtype=float)
    for i in range(14, n):
        if range14[i] > 0:
            chop[i] = 100 * np.log10(atr14[i] / range14[i]) / np.log10(14)
        else:
            chop[i] = 50  # undefined, set to middle
    
    # Align Chop to 1d
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Weekly trend filter (HTF: 1w) ===
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend direction
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    uptrend_1w = close_1w > ema_20_1w
    downtrend_1w = close_1w < ema_20_1w
    # Align weekly trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_20_1w := uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_20_1w := downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 14 for RSI/Chop + 10 for KAMA ER + 1 for seeding)
    start_idx = 25
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price near KAMA (pullback in uptrend), RSI oversold, choppy market, weekly uptrend
            if (close[i] <= kama_aligned[i] * 1.02 and  # within 2% above KAMA
                close[i] >= kama_aligned[i] * 0.98 and  # within 2% below KAMA
                rsi_aligned[i] < 40 and
                chop_aligned[i] > 61.8 and
                uptrend_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price near KAMA (bounce in downtrend), RSI overbought, choppy market, weekly downtrend
            elif (close[i] <= kama_aligned[i] * 1.02 and
                  close[i] >= kama_aligned[i] * 0.98 and
                  rsi_aligned[i] > 60 and
                  chop_aligned[i] > 61.8 and
                  downtrend_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price crosses above KAMA (momentum) OR RSI overbought OR chop drops (trending market)
            if (close[i] > kama_aligned[i] * 1.02 or  # price above KAMA buffer
                rsi_aligned[i] > 70 or
                chop_aligned[i] < 50):  # market trending, mean reversion less effective
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price crosses below KAMA (momentum) OR RSI oversold OR chop drops
            if (close[i] < kama_aligned[i] * 0.98 or  # price below KAMA buffer
                rsi_aligned[i] < 30 or
                chop_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_MeanReversion_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0