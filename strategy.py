#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h KAMA + RSI + Chop Regime
# Hypothesis: KAMA adapts to market noise, RSI identifies momentum extremes, and Chop filter distinguishes trending vs ranging markets.
# In low chop (trending regime), we follow KAMA direction. In high chop (ranging), we mean-revert at RSI extremes.
# This adapts to both bull and bear markets by switching between trend and mean-reversion modes.
# 4h timeframe targets 20-50 trades/year (80-200 over 4 years) to minimize fee drag.
name = "4h_kama_rsi_chop_regime_v1"
timeframe = "4h"
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
    
    # Get 1-day and 1-week data for regime filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) on 4h
    # Efficiency Ratio: ER = |net change| / sum of absolute changes
    change = np.abs(np.diff(close, n=1))
    vol = np.abs(np.diff(close, n=1))
    # Pad change array to match close length
    change_full = np.concatenate([[0], change])
    vol_full = np.concatenate([[0], vol])
    
    # Calculate ER over 10 periods
    change_sum = pd.Series(change_full).rolling(window=10, min_periods=1).sum()
    vol_sum = pd.Series(vol_full).rolling(window=10, min_periods=1).sum()
    er = np.where(vol_sum != 0, change_sum / vol_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) on 4h
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    loss_ma = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = np.where(loss_ma != 0, gain_ma / loss_ma, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop Index (14) - measures choppiness/ranging vs trending
    # High Chop = ranging, Low Chop = trending
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.maximum(np.abs(low - np.roll(close, 1)), tr1)
    tr = np.where(np.arange(len(close)) == 0, high - low, tr2)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14), 
                    50)
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # 1-week trend filter: price above/below weekly EMA(20)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    weekly_ema_4h = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(daily_ema_4h[i]) or np.isnan(weekly_ema_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA cross down OR RSI overbought in ranging market
            if close[i] < kama[i] or (chop[i] > 61.8 and rsi[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: KAMA cross up OR RSI oversold in ranging market
            if close[i] > kama[i] or (chop[i] > 61.8 and rsi[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Trending regime (low chop): follow KAMA with higher timeframe confirmation
                if chop[i] < 38.2:  # Strong trend
                    if close[i] > kama[i] and close[i] > daily_ema_4h[i] and close[i] > weekly_ema_4h[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < kama[i] and close[i] < daily_ema_4h[i] and close[i] < weekly_ema_4h[i]:
                        position = -1
                        signals[i] = -0.25
                # Ranging regime (high chop): mean revert at RSI extremes
                elif chop[i] > 61.8:  # Strong range
                    if rsi[i] < 30:  # Oversold
                        position = 1
                        signals[i] = 0.25
                    elif rsi[i] > 70:  # Overbought
                        position = -1
                        signals[i] = -0.25
                # Transition regime: no trade
    
    return signals