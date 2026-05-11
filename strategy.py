#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Confirmation
# Hypothesis: 1d strategy using KAMA for trend direction, RSI for pullback entries, and volume confirmation.
# Designed for low trade frequency (7-25/year) to minimize fee drag while capturing trend continuations.
# Works in bull via trend-following and in bear via short entries on trend reversals.

name = "1d_KAMA_Trend_RSI_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA trend (10-period ER, 2/30 SC) ---
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily (already aligned as same timeframe)
    kama_aligned = kama
    
    # --- Weekly trend filter: EMA34 ---
    weekly_close = df_1w['close'].values
    ema_34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # --- RSI(14) for pullback entries ---
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad first 14 values
    rsi = np.concatenate([np.full(14, 50), rsi])
    
    # --- Volume confirmation (1.5x 20-day average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: ensure we have enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend alignment: price vs KAMA and weekly EMA
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        weekly_bullish = ema_34_1w_aligned[i] > kama_aligned[i]  # Weekly trend vs KAMA
        weekly_bearish = ema_34_1w_aligned[i] < kama_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above KAMA, weekly bullish, RSI pullback (<40) with volume
            if price_above_kama and weekly_bullish and rsi[i] < 40 and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, weekly bearish, RSI bounce (>60) with volume
            elif price_below_kama and weekly_bearish and rsi[i] > 60 and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit: price crosses below KAMA or weekly trend turns bearish
                if close[i] < kama_aligned[i] or ema_34_1w_aligned[i] < kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: price crosses above KAMA or weekly trend turns bullish
                if close[i] > kama_aligned[i] or ema_34_1w_aligned[i] > kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals