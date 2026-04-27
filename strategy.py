#!/usr/bin/env python3
"""
1d_RelativeStrengthIndex_WeeklyTrend_Filtered
Hypothesis: Weekly trend filtered RSI mean reversion on daily timeframe captures reversals in both bull and bear markets.
Uses weekly trend filter to avoid counter-trend trades, RSI(14) < 30 for long and > 70 for short with volume confirmation.
Target: 15-25 trades/year to minimize fee drag while capturing significant reversals.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI and volume MA
    start_idx = max(35, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema21_1w_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema21_1w_aligned[i]
        rsi_val = rsi[i]
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Long: RSI oversold in uptrend with volume confirmation
            if rsi_val < 30 and close[i] > weekly_trend and vol_confirm_val:
                signals[i] = size
                position = 1
            # Short: RSI overbought in downtrend with volume confirmation
            elif rsi_val > 70 and close[i] < weekly_trend and vol_confirm_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or trend turns down
            if rsi_val > 70 or close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI oversold or trend turns up
            if rsi_val < 30 or close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RelativeStrengthIndex_WeeklyTrend_Filtered"
timeframe = "1d"
leverage = 1.0