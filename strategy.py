#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_With_RSI_And_Chop_Regime_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) as primary trend filter.
Enter long when price > KAMA and RSI < 50 (bullish pullback in uptrend).
Enter short when price < KAMA and RSI > 50 (bearish bounce in downtrend).
Add choppiness regime filter: only trade when CHOP(14) < 61.8 (trending market).
ATR-based stoploss (2.5x) and discrete sizing (0.25).
Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year).
Uses 1w HTF for trend confirmation: only align with weekly trend direction.
Works in bull/bear via adaptive trend following and regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend confirmation)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w EMA34 for weekly trend confirmation ===
    df_1w_close = df_1w['close'].values
    ema_34_1w = pd.Series(df_1w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d OHLC for KAMA, RSI, ATR, and Chop ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA (10, 2, 30) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of abs changes
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = np.concatenate([np.full(10, np.nan), change / volatility])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at first available close after 10 periods
    for i in range(10, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = np.diff(close, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    # Pad first value
    rsi = np.concatenate([[np.nan], rsi])
    
    # === ATR (14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to 0 (no prior close)
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Choppiness Index (14) ===
    # Sum of true range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hh_ll = np.where((hh - ll) == 0, 1e-10, hh - ll)
    chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(14)
    # Pad first 13 values
    chop = np.concatenate([np.full(13, np.nan), chop[13:]])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):  # Start after warmup period
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        atr_val = atr[i]
        chop_val = chop[i]
        weekly_trend = ema_34_1w_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_regime = chop_val < 61.8
        
        if position == 0:
            # Long: price > KAMA (uptrend), RSI < 50 (pullback), weekly trend up
            long_condition = (price > kama_val) and (rsi_val < 50) and (price > weekly_trend) and trending_regime
            # Short: price < KAMA (downtrend), RSI > 50 (bounce), weekly trend down
            short_condition = (price < kama_val) and (rsi_val > 50) and (price < weekly_trend) and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Stoploss (2.5x ATR)
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Trend reversal: price below KAMA
            elif price < kama_val:
                signals[i] = 0.0
                position = 0
            # Weekly trend reversal
            elif price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stoploss (2.5x ATR)
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Trend reversal: price above KAMA
            elif price > kama_val:
                signals[i] = 0.0
                position = 0
            # Weekly trend reversal
            elif price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_Filter_With_RSI_And_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0