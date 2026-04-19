#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1-week RSI trend filter and volume confirmation.
# Long when: price closes above Donchian upper band (20-day high), weekly RSI > 50 (bullish), volume > 1.5x 20-day average
# Short when: price closes below Donchian lower band (20-day low), weekly RSI < 50 (bearish), volume > 1.5x 20-day average
# Exit when price returns to the 20-day midpoint or reverses to opposite band.
# Designed for ~10-20 trades/year per symbol. Works in both bull and bear markets by using weekly RSI for trend direction.
name = "1d_Donchian20_RSI_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate RSI on weekly data (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period]) if not np.isnan(data[1:period]).any() else np.nan
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    rsi_period = 14
    avg_gain = wilders_smoothing(gain, rsi_period)
    avg_loss = wilders_smoothing(loss, rsi_period)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 1d timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Donchian Channels on 1d data (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long breakout: price closes above upper Donchian with weekly RSI > 50 and volume confirmation
            if price > high_20[i] and rsi_val > 50 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below lower Donchian with weekly RSI < 50 and volume confirmation
            elif price < low_20[i] and rsi_val < 50 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to midpoint or breaks below lower band
            if price <= donchian_mid[i] or price < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint or breaks above upper band
            if price >= donchian_mid[i] or price > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals