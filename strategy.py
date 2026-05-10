#!/usr/bin/env python3
# 4h_Contrarian_Volume_Spike_MeanReversion
# Hypothesis: After extreme volume spikes, prices often revert to the mean as liquidity is exhausted.
# This strategy identifies mean-reversion opportunities by combining volume spikes with
# Bollinger Band extremes and RSI conditions. It works in both bull and bear markets
# because it fades momentum exhaustion rather than following trends.

name = "4h_Contrarian_Volume_Spike_MeanReversion"
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
    
    # Volume spike detection (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > volume_ma * 2.0  # Volume at least 2x average
    
    # Bollinger Bands (20-period, 2 std dev)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Bollinger Bands (20) and RSI (14)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Mean reversion conditions
        at_upper_band = close[i] >= bb_upper[i]
        at_lower_band = close[i] <= bb_lower[i]
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        if position == 0:
            # Long entry: price at lower Bollinger Band + RSI oversold + volume spike
            if at_lower_band and rsi_oversold and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price at upper Bollinger Band + RSI overbought + volume spike
            elif at_upper_band and rsi_overbought and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle band or RSI neutral
            if close[i] >= bb_middle[i] or rsi[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle band or RSI neutral
            if close[i] <= bb_middle[i] or rsi[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals