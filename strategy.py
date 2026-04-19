#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA + RSI + chop filter for trend-following in trending markets and mean-reversion in choppy regimes.
# Uses KAMA trend direction for entries, RSI for overbought/oversold confirmation, and Choppiness Index to filter regimes.
# Designed for 12h timeframe with ~15-25 trades/year per symbol (60-100 total over 4 years).
# Works in both bull and bear markets by adapting to regime: trend-follow when trending, mean-revert when choppy.
name = "12h_KAMA_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d, dtype=float)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 0
        else:
            sum_vol = np.sum(volatility[max(1, i-9):i+1])  # 10-period volatility
            er[i] = change[i] / (sum_vol + 1e-10) if sum_vol > 0 else 0
    sc = np.square(er * (2/(2+1) - 1/(30+1)) + 1/(30+1))  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # 1d data for RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d data for Choppiness Index (14-period)
    # Chop = 100 * log10(sum(ATR) / (highest_high - lowest_low)) / log10(period)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(close_1d)
    atr[0] = tr[0]
    for i in range(1, len(close_1d)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Sum of ATR over 14 periods
    sum_atr = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 13:
            sum_atr[i] = np.sum(atr[max(0, i-13):i+1])
        else:
            sum_atr[i] = np.sum(atr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros_like(close_1d)
    lowest_low = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        start_idx = max(0, i-13)
        highest_high[i] = np.max(high_1d[start_idx:i+1])
        lowest_low[i] = np.min(low_1d[start_idx:i+1])
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(sum_atr / range_hl) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: chop > 61.8 = ranging/choppy (mean revert), chop < 38.2 = trending (trend follow)
        is_choppy = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long conditions
            if is_trending:
                # Trend-following: price > KAMA and RSI > 50 (bullish momentum)
                if price > kama_val and rsi_val > 50:
                    signals[i] = 0.25
                    position = 1
            elif is_choppy:
                # Mean reversion: price < KAMA and RSI < 40 (oversold in chop)
                if price < kama_val and rsi_val < 40:
                    signals[i] = 0.25
                    position = 1
            
            # Short conditions
            if is_trending:
                # Trend-following: price < KAMA and RSI < 50 (bearish momentum)
                if price < kama_val and rsi_val < 50:
                    signals[i] = -0.25
                    position = -1
            elif is_choppy:
                # Mean reversion: price > KAMA and RSI > 60 (overbought in chop)
                if price > kama_val and rsi_val > 60:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if is_trending:
                # Exit trend-follow: price < KAMA or RSI < 40
                if price < kama_val or rsi_val < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif is_choppy:
                # Exit mean reversion: price > KAMA or RSI > 60
                if price > kama_val or rsi_val > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # Exit trend-follow: price > KAMA or RSI > 60
                if price > kama_val or rsi_val > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif is_choppy:
                # Exit mean reversion: price < KAMA or RSI < 40
                if price < kama_val or rsi_val < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals