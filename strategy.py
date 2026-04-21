#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend filter with 1d RSI and 12h volume spike.
# Long when KAMA trending up, RSI < 35 (oversold), and volume > 2x 20-period average.
# Short when KAMA trending down, RSI > 65 (overbought), and volume > 2x 20-period average.
# KAMA adapts to market noise, reducing whipsaws in ranging markets.
# Volume spike confirms momentum behind the move.
# Target: 20-40 trades/year by requiring adaptive trend + extreme RSI + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[0:13] = np.nan  # first 13 periods insufficient
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Kaufman's Adaptive Moving Average (KAMA) on 12h data
    close = prices['close'].values
    er_period = 10
    fast_sc = 2 / (2 + 1)  # 2/(fast+1)
    slow_sc = 2 / (30 + 1)  # 2/(slow+1)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=er_period, prepend=close[:er_period]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[er_period-1:], axis=0) if hasattr(np.sum, 'axis') else np.nan
    # Manual volatility sum for efficiency
    volatility = np.zeros_like(close)
    for i in range(er_period, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period+1:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=float)
    kama[er_period] = close[er_period]  # seed
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.where(kama > np.roll(kama, 1), 1, np.where(kama < np.roll(kama, 1), -1, 0))
    kama_dir[0] = 0
    
    # Align KAMA direction to same index (already 12h)
    # No alignment needed as KAMA is calculated on 12h data
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(rsi_aligned[i]) or np.isnan(kama_dir[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long conditions: KAMA up, RSI oversold, volume spike
            if kama_dir[i] == 1 and rsi_aligned[i] < 35 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA down, RSI overbought, volume spike
            elif kama_dir[i] == -1 and rsi_aligned[i] > 65 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if KAMA turns down or RSI becomes overbought
                if kama_dir[i] == -1 or rsi_aligned[i] > 65:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if KAMA turns up or RSI becomes oversold
                if kama_dir[i] == 1 or rsi_aligned[i] < 35:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_KAMA_RSI_Volume_Spike"
timeframe = "12h"
leverage = 1.0