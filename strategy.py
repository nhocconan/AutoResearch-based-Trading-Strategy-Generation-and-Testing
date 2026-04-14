#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day price action with weekly trend filter and volume confirmation
# Long when price closes above weekly 20-period SMA AND price > daily VWAP AND volume > 1.8x 20-day average
# Short when price closes below weekly 20-period SMA AND price < daily VWAP AND volume > 1.8x 20-day average
# Exit when price crosses back below/above weekly 20-period SMA
# Uses weekly SMA for trend alignment, daily VWAP for intraday value, volume for confirmation
# Target: 40-80 total trades over 4 years (10-20/year) to minimize fee drag while capturing trends

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly 20-period SMA for trend filter
    close_1w = df_1w['close'].values
    sma20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    # Calculate daily VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, 0)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for weekly SMA + buffer)
    start = 25
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma20_1w_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.8
        
        if position == 0:
            # Long setup: price above weekly SMA20 AND above VWAP AND volume confirmation
            if (price > sma20_1w_aligned[i] and price > vwap[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price below weekly SMA20 AND below VWAP AND volume confirmation
            elif (price < sma20_1w_aligned[i] and price < vwap[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly SMA20
            if price < sma20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly SMA20
            if price > sma20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklySMA20_VWAP_Volume"
timeframe = "1d"
leverage = 1.0