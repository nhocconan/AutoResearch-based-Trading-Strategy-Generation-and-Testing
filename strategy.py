#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA + RSI + Chop regime - Uses Kaufman Adaptive Moving Average for trend detection
# combined with RSI for momentum and Choppiness Index for regime filtering. Designed for low
# frequency trading (12-37 trades/year) to minimize fee impact. Works in both bull/bear via
# adaptive logic: trend following in trending markets (CHOP < 38.2), mean reversion in ranging
# markets (CHOP > 61.8). Weekly trend filter ensures alignment with higher timeframe direction.

name = "12h_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # KAMA (14-period)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
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
    
    # Choppiness Index (14-period)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.zeros_like(close)
    for i in range(len(close)):
        if max_high[i] != min_low[i] and atr[i] > 0:
            sum_atr = pd.Series(tr).rolling(window=14, min_periods=14).sum().iloc[i]
            chop[i] = 100 * np.log10(sum_atr / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend direction from KAMA
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # Market regime from Choppiness Index
        chop_range = chop[i] > 61.8  # Ranging market
        chop_trend = chop[i] < 38.2  # Trending market
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on reverse signal or at opposite extreme
            if (close[i] < kama[i]) or (rsi[i] > 70 and chop_range):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on reverse signal or at opposite extreme
            if (close[i] > kama[i]) or (rsi[i] < 30 and chop_range):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Ranging market: mean reversion at RSI extremes
            if chop_range:
                # Buy at RSI < 30 with weekly uptrend and volume
                if (rsi[i] < 30) and weekly_uptrend and vol_confirm:
                    position = 1
                    signals[i] = 0.25
                # Sell at RSI > 70 with weekly downtrend and volume
                elif (rsi[i] > 70) and weekly_downtrend and vol_confirm:
                    position = -1
                    signals[i] = -0.25
            # Trending market: trend following with KAMA
            else:
                # Buy when price above KAMA with weekly uptrend and volume
                if kama_up and weekly_uptrend and vol_confirm:
                    position = 1
                    signals[i] = 0.25
                # Sell when price below KAMA with weekly downtrend and volume
                elif kama_down and weekly_downtrend and vol_confirm:
                    position = -1
                    signals[i] = -0.25
    
    return signals