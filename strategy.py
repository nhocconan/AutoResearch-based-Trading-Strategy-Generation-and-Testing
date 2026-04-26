#!/usr/bin/env python3
"""
1h_RSI_MeanReversion_VolumeSpike_SessionFilter
Hypothesis: On 1h timeframe, enter long when RSI(14) < 30 AND volume > 2.0x 20-period average volume AND session is active (08-20 UTC). Enter short when RSI(14) > 70 AND volume > 2.0x 20-period average volume AND session is active. Exit on RSI crossing back to neutral (40-60) or session end. Uses mean reversion in overextended moves with volume confirmation and session filter to reduce noise trades in both bull and bear markets.
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
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (pre-compute once)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI and volume MA warmup
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(rsi[i]) or np.isnan(volume_ma[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Mean reversion conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral_exit = (rsi[i] >= 40) and (rsi[i] <= 60)
        
        if position == 0:
            # Long: RSI oversold + volume spike + in session
            long_signal = rsi_oversold and volume_spike[i] and in_session[i]
            
            # Short: RSI overbought + volume spike + in session
            short_signal = rsi_overbought and volume_spike[i] and in_session[i]
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: RSI back to neutral OR session ends
            if rsi_neutral_exit or not in_session[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: RSI back to neutral OR session ends
            if rsi_neutral_exit or not in_session[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_RSI_MeanReversion_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0