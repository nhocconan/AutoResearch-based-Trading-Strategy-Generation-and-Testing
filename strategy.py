#!/usr/bin/env python3
"""
12h_MultiTF_Confluence_Strategy
Hypothesis: Combine weekly trend bias (price above/below weekly EMA20) with daily momentum (RSI > 50 for long, < 50 for short) and 12h price action (close >/open for long, < for short) to capture medium-term moves. Volatility filter (ATR ratio > 0.8) avoids choppy markets. Designed to work in both bull and bear markets by following the weekly trend. Targets 50-150 total trades over 4 years.
"""

name = "12h_MultiTF_Confluence_Strategy"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for RSI and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- Weekly EMA20 for trend bias ---
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # --- Daily RSI(14) for momentum ---
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # --- Daily ATR(14) for volatility regime ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / (atr_ma_1d + 1e-10)
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 40  # for ATR ratio and other indicators
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_ratio_12h_aligned[i])):
            if position != 0:
                # Simple stoploss: 2.5x ATR from entry
                atr_est = np.abs(high_12h[i] - low_12h[i])  # rough 12h ATR estimate
                if position == 1 and close_12h[i] <= entry_price - 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volatility filter: avoid choppy markets (ATR ratio > 0.8)
        vol_filter = atr_ratio_12h_aligned[i] > 0.8
        
        if position == 0:
            # Look for entries based on multi-timeframe confluence
            if vol_filter:
                # Long conditions: weekly uptrend + daily bullish momentum + 12h bullish candle
                weekly_uptrend = close_12h[i] > ema20_1w_aligned[i]
                daily_bullish = rsi_1d_aligned[i] > 50
                candle_bullish = close_12h[i] > open_12h[i]
                
                if weekly_uptrend and daily_bullish and candle_bullish:
                    signals[i] = 0.25  # long
                    position = 1
                    entry_price = close_12h[i]
                
                # Short conditions: weekly downtrend + daily bearish momentum + 12h bearish candle
                elif (not weekly_uptrend) and (rsi_1d_aligned[i] < 50) and (close_12h[i] < open_12h[i]):
                    signals[i] = -0.25  # short
                    position = -1
                    entry_price = close_12h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit on weekly trend reversal or bearish momentum
                weekly_uptrend = close_12h[i] > ema20_1w_aligned[i]
                daily_bullish = rsi_1d_aligned[i] > 50
                
                if not weekly_uptrend or not daily_bullish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit on weekly trend reversal or bullish momentum
                weekly_downtrend = close_12h[i] < ema20_1w_aligned[i]
                daily_bearish = rsi_1d_aligned[i] < 50
                
                if not weekly_downtrend or not daily_bearish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals