#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_V1
Hypothesis: Daily KAMA trend direction combined with RSI extremes and choppiness regime filter for BTC/ETH.
KAMA adapts to market noise - in trending markets it follows price closely, in ranging markets it stays flat.
Long when KAMA rising AND RSI < 30 (oversold) in choppy or trending regimes.
Short when KAMA falling AND RSI > 70 (overbought) in choppy or trending regimes.
Uses 1-week HTF for higher timeframe trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50.
ATR-based stoploss via signal=0 when price moves against position by 2.5*ATR.
Designed for very low trade frequency (target: 7-25 trades/year) to minimize fee drag and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1 week for trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1-week EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily Indicators (primary timeframe) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Adaptive Moving Average)
    # Efficiency Ratio: |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    er_num = np.abs(np.diff(close, 10, prepend=close[:10]))  # 10-period net change
    er_den = pd.Series(abs_change).rolling(window=10, min_periods=10).sum().values
    er = np.where(er_den != 0, er_num / er_den, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period)
    chop_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(chop_sum / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) 
            or np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Regime detection (choppy or trending both allowed - we adapt)
        is_choppy = chop[i] > 61.8   # mean reversion regime
        is_trending = chop[i] < 38.2  # trend following regime
        
        if position == 0:
            # Long: KAMA rising + RSI oversold + weekly trend filter
            if kama_rising and rsi_oversold and price > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: KAMA falling + RSI overbought + weekly trend filter
            elif kama_falling and rsi_overbought and price < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: KAMA falling or RSI overbought
            elif not kama_rising or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: KAMA rising or RSI oversold
            elif not kama_falling or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_V1"
timeframe = "1d"
leverage = 1.0