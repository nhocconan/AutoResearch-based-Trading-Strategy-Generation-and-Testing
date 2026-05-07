#!/usr/bin/env python3

name = "6h_Retracement_Trend_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for long-term trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 6h ATR(14) for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_filter = atr_14 > 0.5 * atr_ma_50  # Only trade when volatility is above half its 50-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 6h
    
    start_idx = max(60, 50)
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or np.isnan(vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Trend filter: price above/below weekly EMA50
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars and vol_filter[i]:
            # Long: Pullback to weekly EMA50 in uptrend with bullish momentum (RSI > 50)
            if trend_up and close[i] <= ema_50_1w_aligned[i] * 1.01 and rsi_14_1d_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Pullback to weekly EMA50 in downtrend with bearish momentum (RSI < 50)
            elif trend_down and close[i] >= ema_50_1w_aligned[i] * 0.99 and rsi_14_1d_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Trend reversal or overbought RSI
            if not trend_up or rsi_14_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Trend reversal or oversold RSI
            if not trend_down or rsi_14_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: In strong trends (price vs weekly EMA50), retracements to the weekly EMA50 with
# momentum confirmation (daily RSI > 50 for longs, < 50 for shorts) offer high-probability entries.
# The strategy works in bull markets by buying dips to weekly support in uptrends and in bear
# markets by selling rallies to weekly resistance in downtrends. Volatility filter avoids choppy
# markets, and cooldown prevents overtrading. Weekly EMA50 acts as dynamic support/resistance
# that institutions watch, while daily RSI ensures momentum alignment. Target: 15-35 trades/year.