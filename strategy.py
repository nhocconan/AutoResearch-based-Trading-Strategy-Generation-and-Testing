#!/usr/bin/env python3
"""
1d_1w_MultiTimeframe_Momentum_v1
Hypothesis: Combine daily price momentum (close > open) with weekly trend strength (price > weekly EMA50) and volume confirmation. 
This strategy captures strong trending moves while avoiding choppy markets. Designed for low trade frequency (<25 trades/year) 
by requiring confluence of daily momentum, weekly trend, and volume spike. Works in both bull (riding trends) and bear (shorting breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_MultiTimeframe_Momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])  # Simple average for first value
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily volume average (20-period)
    vol_ma20 = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma20[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Daily momentum: close > open (bullish candle) or close < open (bearish candle)
        bullish_momentum = close[i] > open_price[i]
        bearish_momentum = close[i] < open_price[i]
        
        # Weekly trend: price above/below weekly EMA50
        above_weekly_ema = close[i] > ema50_1w_aligned[i]
        below_weekly_ema = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_spike = volume[i] > 1.5 * vol_ma20[i]
        
        # Entry conditions: momentum + trend + volume
        long_entry = bullish_momentum and above_weekly_ema and volume_spike
        short_entry = bearish_momentum and below_weekly_ema and volume_spike
        
        # Exit conditions: opposite momentum or loss of trend
        long_exit = (not bullish_momentum) or (not above_weekly_ema)
        short_exit = (not bearish_momentum) or (not below_weekly_ema)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals