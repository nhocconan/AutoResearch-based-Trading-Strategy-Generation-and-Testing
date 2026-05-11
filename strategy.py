#!/usr/bin/env python3
"""
4h_RSI_Stochastic_Oscillator_12hTrend
Hypothesis: Uses RSI(14) overbought/oversold combined with Stochastic Oscillator for momentum confirmation on 4h, filtered by 12h EMA50 trend. RSI catches reversals, Stochastic confirms momentum shift, and 12h EMA ensures alignment with medium-term trend. Designed for 20-40 trades/year to minimize fee drag while capturing mean reversion in ranging markets and trend continuations in trending markets.
"""

name = "4h_RSI_Stochastic_Oscillator_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # --- 12h Trend Filter: EMA50 ---
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # --- 4h RSI(14) ---
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- 4h Stochastic Oscillator (14,3,3) ---
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close_4h - lowest_low) / (highest_high - lowest_low + 1e-10)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 30  # for RSI and Stochastic
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or np.isnan(d_percent[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_4h[i] <= entry_price - 2.0 * (high_4h[i] - low_4h[i]):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * (high_4h[i] - low_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 12h trend
        trend_up = close_4h[i] > ema50_12h_aligned[i]
        trend_down = close_4h[i] < ema50_12h_aligned[i]
        
        if position == 0:
            # Look for entries: RSI reversal + Stochastic confirmation + trend alignment
            if (rsi[i] < 30 and k_percent[i] > d_percent[i] and trend_up):
                # Long: RSI oversold + Stochastic bullish crossover + 12h uptrend
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif (rsi[i] > 70 and k_percent[i] < d_percent[i] and trend_down):
                # Short: RSI overbought + Stochastic bearish crossover + 12h downtrend
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_4h[i] <= entry_price - 2.0 * (high_4h[i] - low_4h[i]):
                    signals[i] = 0.0
                    position = 0
                # Exit: RSI returns to neutral or Stochastic turns bearish
                elif (rsi[i] >= 50 and k_percent[i] < d_percent[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_4h[i] >= entry_price + 2.0 * (high_4h[i] - low_4h[i]):
                    signals[i] = 0.0
                    position = 0
                # Exit: RSI returns to neutral or Stochastic turns bullish
                elif (rsi[i] <= 50 and k_percent[i] > d_percent[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals