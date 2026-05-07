#!/usr/bin/env python3
name = "6h_MultiTF_Momentum_Aligned_1dTrend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d RSI(14) momentum
    delta = np.diff(df_1d['close'].values)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).ewm(alpha=1/14, adjust=False).values
    roll_down = pd.Series(down).ewm(alpha=1/14, adjust=False).values
    rs = roll_up / (roll_down + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6h Bollinger Bands for mean reversion in ranging markets
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~12 hours for 6h to reduce trades
    
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction and momentum
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        rsi = rsi_1d_aligned[i]
        
        # Determine market regime: trending vs ranging
        bb_width = (bb_upper[i] - bb_lower[i]) / sma_20[i] if sma_20[i] > 0 else 0
        is_trending = bb_width > 0.03  # >3% width indicates trending
        is_ranging = bb_width <= 0.03  # <=3% width indicates ranging
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            if is_trending:
                # Trending regime: follow 1d trend with momentum confirmation
                if trend_up and rsi > 50 and rsi < 70:
                    signals[i] = 0.25
                    position = 1
                    bars_since_last_trade = 0
                elif trend_down and rsi < 50 and rsi > 30:
                    signals[i] = -0.25
                    position = -1
                    bars_since_last_trade = 0
            else:
                # Ranging regime: mean reversion at Bollinger Bands
                if close[i] <= bb_lower[i] and rsi < 40:
                    signals[i] = 0.25
                    position = 1
                    bars_since_last_trade = 0
                elif close[i] >= bb_upper[i] and rsi > 60:
                    signals[i] = -0.25
                    position = -1
                    bars_since_last_trade = 0
        elif position == 1:
            # Exit: trend reversal or overbought conditions
            if not trend_up or rsi >= 70 or close[i] >= bb_upper[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend reversal or oversold conditions
            if not trend_down or rsi <= 30 or close[i] <= bb_lower[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s strategy combining 1d EMA50 trend filter with RSI momentum and 6h Bollinger Bands.
# In trending markets (BB width > 3%): follow 1d trend with RSI confirmation (avoid extremes).
# In ranging markets (BB width <= 3%): mean revert at Bollinger Bands with RSI filtering.
# Uses 6h timeframe to balance trade frequency and capture multi-timeframe alignment.
# Designed to work in both bull and bear markets by adapting to regime conditions.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.