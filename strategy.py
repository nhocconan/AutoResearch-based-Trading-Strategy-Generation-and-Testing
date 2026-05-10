#!/usr/bin/env python3
# 4h_Three_Signal_Confluence
# Hypothesis: Combine Donchian breakout, RSI mean reversion, and volume spike as three independent signals.
# Entry requires all three signals to align, ensuring high-conviction trades.
# Donchian provides trend-following structure, RSI captures overextended reversals, and volume confirms institutional interest.
# This triple confluence reduces false signals and keeps trade frequency low (target: 20-40/year).
# Works in both bull and bear markets by requiring alignment with the primary trend via Donchian,
# while using RSI to enter on pullbacks within the trend.

name = "4h_Three_Signal_Confluence"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_period = 20
    upper_channel = np.zeros(n)
    lower_channel = np.zeros(n)
    for i in range(donchian_period-1, n):
        upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
        lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    # RSI (14-period)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
    for i in range(rsi_period, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike (current > 2.0 * 20-period average)
    vol_ma = np.zeros(n)
    for i in range(20-1, n):
        vol_ma[i] = np.mean(volume[i-20+1:i+1])
    volume_spike = volume > (2.0 * vol_ma)
    
    # Signals
    donchian_breakout_up = close > upper_channel
    donchian_breakout_down = close < lower_channel
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    # Combined entry conditions
    long_entry = donchian_breakout_up & rsi_oversold & volume_spike
    short_entry = donchian_breakout_down & rsi_overbought & volume_spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, rsi_period, 20)
    
    for i in range(start_idx, n):
        if position == 0:
            if long_entry[i]:
                signals[i] = 0.25
                position = 1
            elif short_entry[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on Donchian reversal or RSI overbought
            if donchian_breakout_down[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Donchian reversal or RSI oversold
            if donchian_breakout_up[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals