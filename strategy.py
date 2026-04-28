#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_Filter_VolumeSpike
Hypothesis: Combines Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI for momentum confirmation, and volume spikes to filter entries on 12h timeframe.
KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends.
RSI helps avoid overbought/oversold extremes. Volume spikes confirm institutional interest.
Designed for low trade frequency (12-37/year) to minimize fee drag in both bull and bear markets.
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
    
    # Get 1-day data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day KAMA for trend filter
    close_1d = df_1d['close'].values
    # KAMA parameters: ER length=10, Fast=2, Slow=30
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1-day RSI for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate volume spike (>2.0x 30-period MA for strict filtering)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (2.0 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1-day KAMA
        trend_up = close[i] > kama_1d_aligned[i]
        trend_down = close[i] < kama_1d_aligned[i]
        
        # RSI filters: avoid extremes, look for momentum
        rsi_value = rsi_1d_aligned[i]
        rsi_bullish = 50 < rsi_value < 70  # Not overbought, has upward momentum
        rsi_bearish = 30 < rsi_value < 50  # Not oversold, has downward momentum
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry logic:
        # Long: Uptrend + RSI bullish + volume spike
        long_entry = vol_confirm and trend_up and rsi_bullish
        
        # Short: Downtrend + RSI bearish + volume spike
        short_entry = vol_confirm and trend_down and rsi_bearish
        
        # Exit logic: Opposite signal or RSI extreme
        long_exit = not trend_up or rsi_value >= 70
        short_exit = not trend_down or rsi_value <= 30
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_Trend_RSI_Filter_VolumeSpike"
timeframe = "12h"
leverage = 1.0