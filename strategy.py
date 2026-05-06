#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d RSI for momentum and 1w Donchian channel for trend context
# - Uses 1w Donchian channel (20-period) to establish long-term trend direction
# - Uses 1d RSI (14) for momentum confirmation with overbought/oversold levels
# - Enters long when price is above 1w Donchian upper band and 1d RSI crosses above 50
# - Enters short when price is below 1w Donchian lower band and 1d RSI crosses below 50
# - Exits when price crosses back to the opposite Donchian band or RSI reaches extreme levels
# - Designed to capture trend continuation with momentum confirmation in both bull and bear markets
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_1wDonchian_1dRSI_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Get 1w data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d RSI (14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1w Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band (highest high over 20 periods)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band (lowest low over 20 periods)
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 6h timeframe
    rsi_6h = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Align 1w Donchian channels to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(rsi_6h[i]) or np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 1w Donchian upper band and RSI crosses above 50
            if close[i] > donchian_high_6h[i] and rsi_6h[i] > 50 and rsi_6h[i-1] <= 50:
                signals[i] = 0.25
                position = 1
            # Short: price below 1w Donchian lower band and RSI crosses below 50
            elif close[i] < donchian_low_6h[i] and rsi_6h[i] < 50 and rsi_6h[i-1] >= 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w Donchian lower band or RSI reaches overbought (70)
            if close[i] < donchian_low_6h[i] or rsi_6h[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w Donchian upper band or RSI reaches oversold (30)
            if close[i] > donchian_high_6h[i] or rsi_6h[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals