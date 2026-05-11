#!/usr/bin/env python3
"""
6h_4HourTrend_6HourMomentum
Hypothesis: Use 4h EMA trend filter to determine bias, then enter on 6h momentum bursts with volume confirmation. 
In trending markets (price above/below 4h EMA50), go long/short when 6h RSI crosses 50 with volume > 1.5x average. 
This avoids whipsaws by only taking trades aligned with higher timeframe trend, reducing false signals.
Target: 20-40 trades/year per symbol.
"""

name = "6h_4HourTrend_6HourMomentum"
timeframe = "6h"
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
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 6h data
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 4h EMA50 for trend filter ---
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # --- 6h RSI(14) for momentum ---
    delta = np.diff(close_6h, prepend=close_6h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- 6h Volume average for confirmation ---
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                # Check stoploss (2x ATR from entry)
                atr_est = np.abs(high_6h[i] - low_6h[i])
                if position == 1 and close_6h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine trend bias from 4h EMA50
        bullish_trend = close_6h[i] > ema50_4h_aligned[i]
        bearish_trend = close_6h[i] < ema50_4h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_6h[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for entries
            if bullish_trend and vol_confirm:
                # Long when RSI crosses above 50 in uptrend
                if rsi[i] > 50 and rsi[i-1] <= 50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_6h[i]
            elif bearish_trend and vol_confirm:
                # Short when RSI crosses below 50 in downtrend
                if rsi[i] < 50 and rsi[i-1] >= 50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_6h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position
                # Exit if trend changes or RSI overbought
                if not bullish_trend or rsi[i] >= 70:
                    signals[i] = 0.0
                    position = 0
                # Stoploss: 2x ATR
                elif close_6h[i] <= entry_price - 2.0 * np.abs(high_6h[i] - low_6h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position
                # Exit if trend changes or RSI oversold
                if not bearish_trend or rsi[i] <= 30:
                    signals[i] = 0.0
                    position = 0
                # Stoploss: 2x ATR
                elif close_6h[i] >= entry_price + 2.0 * np.abs(high_6h[i] - low_6h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals