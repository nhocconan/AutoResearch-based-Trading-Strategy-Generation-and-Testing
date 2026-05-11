#!/usr/bin/env python3
"""
1h_Combined_Momentum_Reversion_v1
Hypothesis: Combines 4h trend direction (EMA21) with 1h momentum (RSI) and mean reversion (BB).
Long when: 4h EMA21 up + RSI < 40 + price < BB lower band.
Short when: 4h EMA21 down + RSI > 60 + price > BB upper band.
Uses 4h for trend filter, 1h for entry timing. Targets 15-30 trades/year.
Works in bull/bear by requiring trend alignment and extreme conditions.
"""

name = "1h_Combined_Momentum_Reversion_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- 4h EMA21 for trend ---
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # --- 1h RSI (14) ---
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # --- 1h Bollinger Bands (20, 2) ---
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper = (sma + 2 * std).values
    lower = (sma - 2 * std).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter from 4h
        trend_up = ema_4h_aligned[i] > ema_4h_aligned[i-1]
        trend_down = ema_4h_aligned[i] < ema_4h_aligned[i-1]
        
        if position == 0:
            # Long conditions: uptrend + oversold + below BB lower
            if trend_up and rsi[i] < 40 and close[i] < lower[i]:
                signals[i] = 0.20
                position = 1
            # Short conditions: downtrend + overbought + above BB upper
            elif trend_down and rsi[i] > 60 and close[i] > upper[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: trend turns down OR RSI > 60 OR price > BB middle
                exit = (not trend_up) or (rsi[i] > 60) or (close[i] > sma.iloc[i])
                if exit:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: trend turns up OR RSI < 40 OR price < BB middle
                exit = (not trend_down) or (rsi[i] < 40) or (close[i] < sma.iloc[i])
                if exit:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals