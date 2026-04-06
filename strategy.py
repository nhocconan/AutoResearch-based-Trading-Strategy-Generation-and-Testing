# 1/1/100
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 1-day EMA trend filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets, EMA filter ensures alignment with daily trend,
# volume confirmation avoids false breakouts. Designed for 4h timeframe to target 75-200 trades over 4 years
# with moderate frequency, suitable for both bull and bear markets via breakout symmetry.

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4-hour Donchian Channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # 1-day EMA (50-period) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            ema_1d[i] = close_1d[i]
        else:
            ema_1d[i] = 0.02 * close_1d[i] + 0.98 * ema_1d[i-1]
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-day volume average (20-period) for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    
    for i in range(len(vol_1d)):
        if i < 19:
            vol_ma_1d[i] = np.nan
        else:
            vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of Donchian 19, EMA 0, vol MA 19)
    start = 19
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.2x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.2
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below Donchian lower band or stoploss
            if (close[i] < lowest_low[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian upper band or stoploss
            if (close[i] > highest_high[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume and trend filter
            if volume_filter:
                # Long: price breaks above upper band and above daily EMA
                if close[i] > highest_high[i] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below lower band and below daily EMA
                elif close[i] < lowest_low[i] and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals