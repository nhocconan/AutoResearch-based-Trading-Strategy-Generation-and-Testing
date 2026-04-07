#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly KAMA with RSI Filter and Volume Confirmation
# Hypothesis: On daily timeframe, use weekly Kaufman Adaptive Moving Average (KAMA) 
# to determine trend direction, combined with daily RSI for mean-reversion entries
# and volume confirmation. In bull markets: buy when price pulls back to KAMA in uptrend
# with RSI < 40. In bear markets: sell when price rallies to KAMA in downtrend
# with RSI > 60. Volume > 1.5x average confirms institutional interest.
# Target: 10-25 trades/year (40-100 over 4 years).

name = "1d_weekly_kama_rsi_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA ( Kaufman Adaptive Moving Average )
    weekly_close = df_weekly['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(weekly_close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(weekly_close, n=1)), axis=1)  # sum |close[i] - close[i-1]| over 10 periods
    # Fix dimensions: change has len-10, volatility has len-1
    # We'll compute ER for indices where both are available
    er = np.zeros_like(weekly_close)
    for i in range(10, len(weekly_close)):
        if np.sum(np.abs(np.diff(weekly_close[i-9:i+1], n=1))) > 0:
            er[i] = np.abs(weekly_close[i] - weekly_close[i-10]) / np.sum(np.abs(np.diff(weekly_close[i-9:i+1], n=1)))
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(weekly_close)
    kama[0] = weekly_close[0]
    for i in range(1, len(weekly_close)):
        kama[i] = kama[i-1] + sc[i] * (weekly_close[i] - kama[i-1])
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    kama = np.roll(kama, 1)
    if len(kama) > 1:
        kama[0] = kama[1]
    else:
        kama[0] = weekly_close[0] if len(weekly_close) > 0 else 0
    
    # Align to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_weekly, kama)
    
    # Daily RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or RSI > 70 (overbought)
            if close[i] < kama_aligned[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or RSI < 30 (oversold)
            if close[i] > kama_aligned[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price near KAMA from below, RSI < 40, volume confirmation
            if (close[i] > kama_aligned[i] * 0.995 and close[i] < kama_aligned[i] * 1.005 and
                rsi[i] < 40 and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price near KAMA from above, RSI > 60, volume confirmation
            elif (close[i] < kama_aligned[i] * 1.005 and close[i] > kama_aligned[i] * 0.995 and
                  rsi[i] > 60 and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals