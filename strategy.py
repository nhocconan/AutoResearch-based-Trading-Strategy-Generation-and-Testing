#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI with 4-hour trend filter and volume confirmation
# Long when RSI(14) > 55, price > 4h EMA200, and volume > 1.5x 20-period average
# Short when RSI(14) < 45, price < 4h EMA200, and volume > 1.5x 20-period average
# Exit when RSI crosses back to neutral (45-55) or volume drops
# Uses 4h for trend direction, 1h for entry timing, volume for confirmation
# Target: 80-150 total trades over 4 years (20-38/year) with strict entry conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4-hour data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate RSI on 1h (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 200  # Wait for 4h EMA200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: RSI > 55, price > 4h EMA200, volume confirmation
            if (rsi[i] > 55 and price > ema200_4h_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: RSI < 45, price < 4h EMA200, volume confirmation
            elif (rsi[i] < 45 and price < ema200_4h_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI drops below 50 or volume drops
            if rsi[i] < 50 or vol < vol_avg[i] * 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI rises above 50 or volume drops
            if rsi[i] > 50 or vol < vol_avg[i] * 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_RSI_4hEMA200_Volume"
timeframe = "1h"
leverage = 1.0