#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_12h_kama_rsi_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Calculate 14-period KAMA on weekly close
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1w, np.nan)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # KAMA trend: 1 if close > KAMA, -1 if close < KAMA
    kama_trend = np.where(close_1w > kama, 1, -1)
    
    # Align weekly KAMA trend to daily
    kama_trend_aligned = align_htf_to_ltf(prices, df_1w, kama_trend)
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Calculate 20-period average volume on 12h
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h volume MA to daily
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_trend_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current daily volume > 1.5x 12h volume MA
        vol_confirm = volume_current > 1.5 * vol_ma_20_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Trend filter from weekly KAMA
        uptrend = kama_trend_aligned[i] == 1
        downtrend = kama_trend_aligned[i] == -1
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: RSI oversold + uptrend + volume confirmation
        if rsi_oversold and uptrend and vol_confirm:
            enter_long = True
        
        # Short: RSI overbought + downtrend + volume confirmation
        if rsi_overbought and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: RSI returns to neutral zone (40-60)
        exit_long = rsi[i] > 40
        exit_short = rsi[i] < 60
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily KAMA + RSI breakout with weekly trend filter and 12h volume confirmation.
# Uses weekly KAMA for trend direction (avoids counter-trend trades in strong trends).
# Enters long when daily RSI < 30 (oversold) in weekly uptrend with volume > 1.5x 12h average.
# Enters short when daily RSI > 70 (overbought) in weekly downtrend with volume > 1.5x 12h average.
# Exits when RSI returns to neutral zone (40 for longs, 60 for shorts).
# Volume confirmation from 12h timeframe reduces false signals.
# Position size 0.25 balances risk and return.
# Target: 10-20 trades per year (40-80 total over 4 years) to minimize fee drag.
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
# Avoids whipsaws by requiring trend alignment and volume confirmation.