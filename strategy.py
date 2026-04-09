#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d RSI extremes with 12h volume spike confirmation
# RSI(14) < 30 = oversold, RSI(14) > 70 = overbought
# Volume spike = current volume > 1.5 * 20-period average volume
# Long when RSI < 30 + volume spike, Short when RSI > 70 + volume spike
# Exit when RSI returns to 50 (neutral)
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Works in bull/bear markets: mean reversion at extremes with volume confirmation filters false signals

name = "6h_12h_1d_rsi_volume_spike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        def wilders_smoothing(values, period):
            if len(values) < period:
                return np.full(len(values), np.nan)
            alpha = 1.0 / period
            result = np.full(len(values), np.nan)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            return result
        
        avg_gain = wilders_smoothing(gain, period)
        avg_loss = wilders_smoothing(loss, period)
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Load 12h data for volume spike confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume moving average (20-period)
    def calculate_sma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        for i in range(period-1, len(values)):
            result[i] = np.mean(values[i-period+1:i+1])
        return result
    
    vol_ma_12h = calculate_sma(volume_12h, 20)
    volume_spike_12h = np.where(vol_ma_12h > 0, volume_12h / vol_ma_12h, 1.0)
    
    # Align indicators to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_spike_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long when RSI returns to neutral (50)
            if rsi_1d_aligned[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when RSI returns to neutral (50)
            if rsi_1d_aligned[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when RSI oversold (<30) + volume spike (>1.5)
            if rsi_1d_aligned[i] < 30 and volume_spike_12h_aligned[i] > 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short when RSI overbought (>70) + volume spike (>1.5)
            elif rsi_1d_aligned[i] > 70 and volume_spike_12h_aligned[i] > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals