#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation
# Uses 1h RSI(14) for entry timing, 4h EMA(50) for trend direction, and volume spike for confirmation.
# Long when RSI < 30 (oversold) in uptrend, short when RSI > 70 (overbought) in downtrend.
# Designed for low trade frequency in both bull and bear markets.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.

name = "1h_RSI14_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_4h_val = ema50_4h_aligned[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: RSI < 30 (oversold) + uptrend + volume spike
            if (rsi_val < 30 and 
                close[i] > ema50_4h_val and 
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Enter short: RSI > 70 (overbought) + downtrend + volume spike
            elif (rsi_val > 70 and 
                  close[i] < ema50_4h_val and 
                  vol_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (normal) OR price breaks below trend
            if rsi_val > 50 or close[i] < ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 50 (normal) OR price breaks above trend
            if rsi_val < 50 or close[i] > ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals