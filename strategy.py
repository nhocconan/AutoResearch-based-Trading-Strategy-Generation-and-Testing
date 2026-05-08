#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d RSI(14) filter + volume confirmation
# Long when price breaks above 20-period high, RSI < 60 (not overbought), volume > 1.5x average
# Short when price breaks below 20-period low, RSI > 40 (not oversold), volume > 1.5x average
# Uses Donchian for trend capture, RSI to avoid overextended entries, volume for confirmation
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_Donchian20_1dRSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    daily_close = df_1d['close'].values
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: break above Donchian high, RSI < 60, volume > 1.5x average
            if close[i] > donch_high[i] and rsi_1d_aligned[i] < 60 and volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low, RSI > 40, volume > 1.5x average
            elif close[i] < donch_low[i] and rsi_1d_aligned[i] > 40 and volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below Donchian low or RSI > 70 (overbought)
            if close[i] < donch_low[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above Donchian high or RSI < 30 (oversold)
            if close[i] > donch_high[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals