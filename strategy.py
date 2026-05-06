#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Supertrend for trend direction and 1-day RSI for mean reversion
# - Uses 1-day Supertrend (ATR=10, multiplier=3) to identify primary trend
# - Enters long when price dips below 1-day RSI 30 in an uptrend (mean reversion long)
# - Enters short when price rises above 1-day RSI 70 in a downtrend (mean reversion short)
# - Exits when price returns to 1-day RSI 50 (mean reversion complete)
# - Designed to work in ranging markets (RSI mean reversion) and trending markets (Supertrend filter)
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_Supertrend1d_RSI14_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Supertrend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1-day ATR for Supertrend
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend upper and lower bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3 * atr_1d)
    lower_band = hl2 - (3 * atr_1d)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_1d[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = upper_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = lower_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Calculate 1-day RSI
    delta = np.diff(close_1d)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align Supertrend direction and RSI to 6h timeframe
    supertrend_direction_6h = align_htf_to_ltf(prices, df_1d, direction)
    rsi_1d_6h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma_20)
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(supertrend_direction_6h[i]) or np.isnan(rsi_1d_6h[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI < 30 in uptrend (mean reversion long)
            if rsi_1d_6h[i] < 30 and supertrend_direction_6h[i] == 1 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 in downtrend (mean reversion short)
            elif rsi_1d_6h[i] > 70 and supertrend_direction_6h[i] == -1 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to 50 (mean reversion complete)
            if rsi_1d_6h[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to 50 (mean reversion complete)
            if rsi_1d_6h[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals