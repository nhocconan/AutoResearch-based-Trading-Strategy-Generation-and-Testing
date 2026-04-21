#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Candlestick Range Breakout with 1d EMA Trend Filter and Volume Confirmation.
# Uses 1d EMA(34) to determine trend direction, enters long when price breaks above 4h range + volume,
# short when breaks below 4h range + volume in downtrend. Avoids whipsaws by requiring trend alignment.
# Target: 20-35 trades/year by requiring strong breakout with volume and trend confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate 4h range (high-low of current bar)
    range_high = prices['high'].values
    range_low = prices['low'].values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Get 1d EMA(34) for trend filter - calculated once, aligned to 4h
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d)
    ema_1d[0] = close_1d[0]
    alpha = 2 / (34 + 1)
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA to 4h timeframe (waits for 1d bar to close)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if data not ready
        if np.isnan(vol_ma[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: use 1d EMA(34)
        trend_up = ema_1d_aligned[i] > ema_1d_aligned[i-1]  # EMA rising
        trend_down = ema_1d_aligned[i] < ema_1d_aligned[i-1]  # EMA falling
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above 4h range in uptrend
                if price > range_high[i] and trend_up:
                    signals[i] = 0.30
                    position = 1
                # Short: price breaks below 4h range in downtrend
                elif price < range_low[i] and trend_down:
                    signals[i] = -0.30
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on trend reversal or range breakdown
                if not trend_up or price < range_low[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on trend reversal or range breakout
                if not trend_down or price > range_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_RangeBreakout_1dEMA34Trend_Volume"
timeframe = "4h"
leverage = 1.0