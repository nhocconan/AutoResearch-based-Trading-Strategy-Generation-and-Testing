# 1d_Pivot_R1_S1_Breakout_Volume_1wRSI_Filter_v1
# Hypothesis: Daily Camarilla pivot levels (R1/S1) act as key support/resistance. 
# Buy when price breaks above R1 with volume confirmation and weekly RSI < 60 (avoid overbought).
# Sell when price breaks below S1 with volume confirmation and weekly RSI > 40 (avoid oversold).
# Uses 1d timeframe for signal generation, 1w for regime filter. Designed for ~15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    n = len(high)
    pivot = (high + low + close) / 3
    range_ = high - low
    r1 = close + range_ * 1.1 / 12
    s1 = close - range_ * 1.1 / 12
    return pivot, r1, s1

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full(len(close), np.nan)
    rsi = np.full(len(close), np.nan)
    
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivots on daily
    _, r1_1d, s1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Get weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    rsi_14_1w = calculate_rsi(close_1w, 14)
    
    # Align to daily timeframe (since we're using 1d timeframe)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Volume confirmation: 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(rsi_14_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1, RSI not overbought, volume confirmation
            if close[i] > r1_1d_aligned[i] and rsi_14_1w_aligned[i] < 60 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, RSI not oversold, volume confirmation
            elif close[i] < s1_1d_aligned[i] and rsi_14_1w_aligned[i] > 40 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 or RSI becomes overbought
            if close[i] < s1_1d_aligned[i] or rsi_14_1w_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 or RSI becomes oversold
            if close[i] > r1_1d_aligned[i] or rsi_14_1w_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Pivot_R1_S1_Breakout_Volume_1wRSI_Filter_v1"
timeframe = "1d"
leverage = 1.0