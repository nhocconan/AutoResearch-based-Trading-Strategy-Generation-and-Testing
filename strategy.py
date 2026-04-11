#!/usr/bin/env python3
# 12h_1w_kama_rsi_volatility_breakout_v1
# Strategy: 12h KAMA trend direction with RSI momentum filter and volatility filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market efficiency, providing reliable trend direction in both trending and ranging markets.
# Combined with RSI for momentum confirmation and volatility filter to avoid low-volatility chop.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Designed for low trade frequency (<30/year) to minimize fee drag and survive bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_kama_rsi_volatility_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h KAMA calculation
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing Constant
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_values = kama(close, 10, 2, 30)
    
    # 12h RSI calculation
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_values = rsi(close, 14)
    
    # 12h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(kama_values[i]) or np.isnan(rsi_values[i]) or np.isnan(atr[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: avoid low volatility (range) markets
        # ATR ratio: current ATR vs 50-period average ATR
        if i >= 50:
            atr_ma = pd.Series(atr[:i+1]).rolling(window=50, min_periods=50).mean().iloc[-1]
            vol_filter = atr[i] > 0.8 * atr_ma  # Only trade when volatility is above 80% of average
        else:
            vol_filter = True
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Trend filter: price vs KAMA and weekly EMA
        price_above_kama = close[i] > kama_values[i]
        price_below_kama = close[i] < kama_values[i]
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi_values[i] < 70
        rsi_not_oversold = rsi_values[i] > 30
        
        # Entry conditions
        # Long: Price crosses above KAMA AND weekly uptrend AND volume confirmation AND volatility filter AND RSI not overbought
        if price_above_kama and weekly_uptrend and vol_confirm and vol_filter and rsi_not_overbought and position != 1:
            # Additional check: ensure we didn't just cross above KAMA in previous bar
            if i == 50 or close[i-1] <= kama_values[i-1]:
                position = 1
                signals[i] = 0.25
        # Short: Price crosses below KAMA AND weekly downtrend AND volume confirmation AND volatility filter AND RSI not oversold
        elif price_below_kama and weekly_downtrend and vol_confirm and vol_filter and rsi_not_oversold and position != -1:
            # Additional check: ensure we didn't just cross below KAMA in previous bar
            if i == 50 or close[i-1] >= kama_values[i-1]:
                position = -1
                signals[i] = -0.25
        # Exit: Price crosses back through KAMA (mean reversion signal)
        elif position == 1 and price_below_kama:
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_above_kama:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals