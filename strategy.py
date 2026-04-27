# Hypothetical: 6h momentum with 1-week RSI divergence + volume confirmation
# Long when: 1-week RSI > 70 and making lower high (bearish divergence) + price breaks above 6h high of last 20 bars + volume > 1.5x average
# Short when: 1-week RSI < 30 and making higher low (bullish divergence) + price breaks below 6h low of last 20 bars + volume > 1.5x average
# Exit when RSI returns to neutral zone (40-60) or opposite divergence appears
# Designed for 6h timeframe: targets 50-150 total trades over 4 years (12-37/year).

#!/usr/bin/env python3

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
    
    # Get 1w data for RSI divergence
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 1-week RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # 6h high/low of last 20 bars for breakout
    high_max_20 = np.full(n, np.nan, dtype=np.float64)
    low_min_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        high_max_20[i] = np.max(high[i-19:i+1])
        low_min_20[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1w RSI (14 periods), 6h high/low (20 periods), volume MA (20 periods)
    start_idx = max(14, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        rsi = rsi_1w_aligned[i]
        high_max = high_max_20[i]
        low_min = low_min_20[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Divergence detection (simplified: using prior bar RSI)
        if i >= start_idx + 1:
            prev_rsi = rsi_1w_aligned[i-1]
            # Bearish divergence: RSI > 70 and making lower high
            bear_div = (rsi > 70) and (prev_rsi > 70) and (rsi < prev_rsi)
            # Bullish divergence: RSI < 30 and making higher low
            bull_div = (rsi < 30) and (prev_rsi < 30) and (rsi > prev_rsi)
        else:
            bear_div = False
            bull_div = False
        
        if position == 0:
            # Long: bullish divergence + price breaks above 6h high + volume spike
            if bull_div and price > high_max and vol_filter:
                signals[i] = size
                position = 1
            # Short: bearish divergence + price breaks below 6h low + volume spike
            elif bear_div and price < low_min and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or bearish divergence appears
            if (rsi >= 40 and rsi <= 60) or bear_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or bullish divergence appears
            if (rsi >= 40 and rsi <= 60) or bull_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI_Divergence_1w_Breakout_Volume"
timeframe = "6h"
leverage = 1.0