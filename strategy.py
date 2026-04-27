#!/usr/bin/env python3
"""
4h_RSI_Overbought_Oversold_12hTrend_Volume
Hypothesis: Combines RSI extremes (RSI<30 for long, RSI>70 for short) with 12h EMA50 trend and volume spike (>1.5x 20-period average) for high-probability mean reversion entries. Designed for low trade frequency (~20-30 trades/year) to minimize fee drift, effective in both bull and bear markets by fading extremes in the direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h EMA50 for trend confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI and volume
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema50_val = ema50_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: RSI oversold (<30), above EMA50 trend, volume confirmation
            if rsi_val < 30 and close[i] > ema50_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: RSI overbought (>70), below EMA50 trend, volume confirmation
            elif rsi_val > 70 and close[i] < ema50_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (>50) or price below EMA50
            if rsi_val > 50 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (<50) or price above EMA50
            if rsi_val < 50 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSI_Overbought_Oversold_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0