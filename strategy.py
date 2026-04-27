# %pip install pandas numpy
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w RSI(14) trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel with 1w RSI > 50 and volume > 1.5x average.
# Short when price breaks below Donchian lower channel with 1w RSI < 50 and volume > 1.5x average.
# Exit when price crosses back through Donchian midpoint.
# Uses weekly RSI to filter breakouts in both bull and bear markets, targeting 12-37 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w RSI(14)
    rsi_period = 14
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1w), np.nan)
    avg_loss = np.full(len(close_1w), np.nan)
    
    # Initial average
    if len(gain) >= rsi_period:
        avg_gain[rsi_period - 1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period - 1] = np.mean(loss[:rsi_period])
        
        for i in range(rsi_period, len(close_1w)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1w = np.full(len(close_1w), np.nan)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align 1w RSI to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate Donchian channels (20-period)
    donchian_period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        highest_high[i] = np.max(high[i - donchian_period + 1:i + 1])
        lowest_low[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Calculate Donchian midpoint
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, RSI, and volume MA20
    start_idx = max(donchian_period - 1, rsi_period, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above Donchian upper with 1w RSI > 50 and volume filter
            if (price > highest_high[i] and 
                rsi_1w_aligned[i] > 50 and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian lower with 1w RSI < 50 and volume filter
            elif (price < lowest_low[i] and 
                  rsi_1w_aligned[i] < 50 and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if price < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if price > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1wRSI_Volume"
timeframe = "6h"
leverage = 1.0