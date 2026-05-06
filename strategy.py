#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and RSI filter
# - Uses 1w Donchian channels (20-period) for long-term structure
# - Uses 1d volume spike for entry confirmation
# - Uses 1d RSI > 60 for long and < 40 for short to filter momentum
# - Enters long when price breaks above 1w Donchian upper band with volume and RSI > 60
# - Enters short when price breaks below 1w Donchian lower band with volume and RSI < 40
# - Exits when price returns to 1w Donchian middle (median)
# - Designed to capture major trend moves with institutional level respect
# - Target: 40-100 total trades over 4 years (10-25/year) with 0.25 position sizing

name = "1d_1wDonchian_20_Volume_RSI"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian upper and lower bands
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2  # Median line for exit
    
    # Align 1w Donchian channels to 1d timeframe
    upper_20_1d = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_1d = align_htf_to_ltf(prices, df_1w, lower_20)
    middle_20_1d = align_htf_to_ltf(prices, df_1w, middle_20)
    
    # Volume filter (1d timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.8 * vol_ma_10)  # Strong volume confirmation
    
    # RSI filter (1d timeframe)
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        avg_gain[period-1] = np.mean(gain[1:period+1])
        avg_loss[period-1] = np.mean(loss[1:period+1])
        
        for i in range(period, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.zeros_like(close_prices)
        rsi = np.zeros_like(close_prices)
        for i in range(period, len(close_prices)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100
        return rsi
    
    rsi_values = calculate_rsi(close, 14)
    rsi_long_filter = rsi_values > 60  # Strong momentum for long
    rsi_short_filter = rsi_values < 40  # Weak momentum for short
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_20_1d[i]) or np.isnan(lower_20_1d[i]) or 
            np.isnan(middle_20_1d[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(rsi_long_filter[i]) or np.isnan(rsi_short_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 1w Donchian upper with volume and RSI > 60
            if close[i] > upper_20_1d[i] and volume_spike[i] and rsi_long_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 1w Donchian lower with volume and RSI < 40
            elif close[i] < lower_20_1d[i] and volume_spike[i] and rsi_short_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle
            if close[i] < middle_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle
            if close[i] > middle_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals