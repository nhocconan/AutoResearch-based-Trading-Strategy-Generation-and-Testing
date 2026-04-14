#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA with weekly trend filter and volume confirmation.
# KAMA adapts to market conditions - fast in trends, slow in ranges.
# Weekly trend filter ensures we only trade in direction of higher timeframe trend.
# Volume confirmation filters out low-conviction moves.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA34 for trend direction (more stable than price vs EMA)
    ema_len = 34
    if len(df_1w) < ema_len:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily KAMA (adaptive moving average)
    er_len = 10      # Efficiency ratio period
    fast_sc = 2 / (2 + 1)   # SC for fastest EMA
    slow_sc = 2 / (30 + 1)  # SC for slowest EMA
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, k=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    
    # Pad arrays to match length
    change = np.concatenate([np.full(er_len, np.nan), change])
    volatility = np.concatenate([np.full(er_len, np.nan), volatility])
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Calculate smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]  # Seed with first close
    
    for i in range(er_len + 1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Daily RSI(14) for overbought/oversold conditions
    rsi_len = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(rsi_len, np.nan), rsi])
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(ema_len*2, er_len*2, rsi_len*2, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA34
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above KAMA, weekly uptrend, not overbought, with volume
            if (close[i] > kama[i] and 
                weekly_uptrend and 
                rsi[i] < 70 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price below KAMA, weekly downtrend, not oversold, with volume
            elif (close[i] < kama[i] and 
                  weekly_downtrend and 
                  rsi[i] > 30 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below KAMA or overbought
            if close[i] < kama[i] or rsi[i] > 75:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price above KAMA or oversold
            if close[i] > kama[i] or rsi[i] < 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "daily_kama_weekly_trend_volume_filter_v1"
timeframe = "1d"
leverage = 1.0