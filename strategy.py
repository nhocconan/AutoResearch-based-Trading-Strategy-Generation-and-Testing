#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and ATR filter
# Long when price breaks above 1d Donchian upper channel (20-period) AND 12h volume > 1.5 * avg_volume(20) AND ATR(14) > 0.01 * close
# Short when price breaks below 1d Donchian lower channel (20-period) AND 12h volume > 1.5 * avg_volume(20) AND ATR(14) > 0.01 * close
# Exit when price crosses the 12h EMA(50) in opposite direction
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Donchian channel from 1d provides structural breakout levels
# Volume spike confirms institutional participation
# ATR filter ensures sufficient volatility to avoid choppy markets
# Works in both bull (continuation breakouts) and bear (continuation breakdowns) markets

name = "12h_1dDonchian20_Volume_ATR_Filter_EMA50_Exit"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for Donchian
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channel (20-period)
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 12h timeframe (wait for completed 1d bar)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate ATR(14) on 12h for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First value
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_filter = atr_14 > (0.01 * close)  # ATR > 1% of price
    
    # Calculate 12h EMA(50) for exit signal
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(atr_14[i]) or np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper channel with volume spike and sufficient volatility
            if (close[i] > upper_aligned[i] and close[i-1] <= upper_aligned[i-1] and 
                volume_confirm[i] and atr_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower channel with volume spike and sufficient volatility
            elif (close[i] < lower_aligned[i] and close[i-1] >= lower_aligned[i-1] and 
                  volume_confirm[i] and atr_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA(50)
            if close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA(50)
            if close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals