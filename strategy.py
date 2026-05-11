#!/usr/bin/env python3
"""
4H_RSI_Extremes_12hTrend_VolumeFilter
Hypothesis: RSI extremes (overbought/oversold) combined with 12h trend filter and volume confirmation work in both bull and bear markets.
In bull markets, buy when RSI < 30 (oversold) with 12h uptrend and volume spike.
In bear markets, sell when RSI > 70 (overbought) with 12h downtrend and volume spike.
Volume filter ensures institutional participation. 4h timeframe balances trade frequency and cost.
Target: 25-40 trades/year to stay under 100 total trades over 4 years, minimizing fee drag.
"""

name = "4H_RSI_Extremes_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend - using close prices
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # RSI calculation on 4h closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing: alpha = 1/period
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema12h = close[i] > ema50_12h_aligned[i]
        price_below_ema12h = close[i] < ema50_12h_aligned[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # Long: RSI oversold + above 12h EMA50 + volume spike
            if rsi_oversold and price_above_ema12h and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: RSI overbought + below 12h EMA50 + volume spike
            elif rsi_overbought and price_below_ema12h and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - RSI mean reversion or trend change
            if position == 1:
                # Exit: RSI returns to neutral (>=50) OR trend turns bearish
                if rsi[i] >= 50 or close[i] < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: RSI returns to neutral (<=50) OR trend turns bullish
                if rsi[i] <= 50 or close[i] > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals