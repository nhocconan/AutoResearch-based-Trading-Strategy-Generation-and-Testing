#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Bollinger Band breakout with daily volume confirmation and RSI filter
# Long when price breaks above weekly upper BB AND daily RSI < 70 AND volume > 1.5x 20-day avg
# Short when price breaks below weekly lower BB AND daily RSI > 30 AND volume > 1.5x 20-day avg
# Exit when price crosses back inside the weekly Bollinger Bands
# Uses weekly structure for trend context, daily for entry timing and volume confirmation
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    
    # Load daily data ONCE before loop for RSI and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly Bollinger Bands (20-period, 2 std)
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    
    # Calculate daily RSI (14-period)
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Calculate daily volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of weekly 20, daily 14, daily 20)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg_aligned[i] * 1.5
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Long setup: breakout above weekly upper BB + RSI not overbought + volume confirmation
            if (price > upper_bb_aligned[i] and rsi_val < 70 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakout below weekly lower BB + RSI not oversold + volume confirmation
            elif (price < lower_bb_aligned[i] and rsi_val > 30 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back inside weekly Bollinger Bands (below upper band)
            if price < upper_bb_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back inside weekly Bollinger Bands (above lower band)
            if price > lower_bb_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyBB_RSI_Volume"
timeframe = "1d"
leverage = 1.0