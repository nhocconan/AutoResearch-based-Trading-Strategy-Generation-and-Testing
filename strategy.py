#!/usr/bin/env python3
"""
4h_Triple_Threat_Reversal
Hypothesis: Combine three independent signals for high-probability reversals:
1. RSI(2) extreme (<10 for long, >90 for short) for mean reversion
2. Bullish/Bearish Engulfing candle pattern for price action confirmation
3. Volume spike (>2x 20-period average) for institutional participation
Works in bull/bear by capturing overextended moves likely to reverse.
Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "4h_Triple_Threat_Reversal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(2) for extreme mean reversion signals
    def rsi(close, period):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        return 100 - (100 / (1 + rs))
    
    rsi_values = rsi(close, 2)
    
    # Engulfing patterns
    bullish_engulfing = (close > open_) & (open_ < close) & (close >= open_.shift(1)) & (open_ <= close.shift(1))
    bearish_engulfing = (close < open_) & (open_ > close) & (open_ >= close.shift(1)) & (close <= open_.shift(1))
    # Fix: need open prices
    open_ = prices['open'].values
    bullish_engulfing = (close > open_) & (open_ < close.shift(1)) & (close >= open_.shift(1)) & (open_ <= close.shift(1))
    bearish_engulfing = (close < open_) & (open_ > close.shift(1)) & (open_ >= close.shift(1)) & (close <= open_.shift(1))
    
    # Volume spike (>2x 20-period average)
    vol_mean = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_mean[i] = np.mean(volume[i-20:i])
    volume_spike = volume > 2 * vol_mean
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume average
    
    for i in range(start_idx, n):
        # Skip if not enough data for calculations
        if i < 2:  # RSI needs at least 2 periods
            continue
            
        # Entry conditions
        if position == 0:
            # Long: RSI(2) < 10, bullish engulfing, volume spike
            if rsi_values[i] < 10 and bullish_engulfing[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90, bearish engulfing, volume spike
            elif rsi_values[i] > 90 and bearish_engulfing[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral (>30) or opposite engulfing
            if rsi_values[i] > 30 or bearish_engulfing[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral (<70) or opposite engulfing
            if rsi_values[i] < 70 or bullish_engulfing[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals