#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian Breakout with Volume Confirmation and RSI Filter
# Uses 1-week Donchian channels for major trend structure, entered only on high volume
# and confirmed by RSI not being overextended. Designed for low turnover (10-30 trades/year)
# to minimize fee drag while capturing strong trending moves in both bull and bear markets.
# Weekly timeframe reduces noise, daily timeframe provides timely entries.
name = "1d_WeeklyDonchian20_Volume_RSI"
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
    
    # Weekly Donchian Channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period high and low for weekly Donchian
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (waits for weekly candle to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    # RSI filter: avoid overextended conditions
    def calculate_rsi(data, period=14):
        delta = np.diff(data, prepend=data[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        
        # First average
        if len(data) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            # Wilder smoothing
            for i in range(period, len(data)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    rsi_not_overbought = rsi < 70
    rsi_not_oversold = rsi > 30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + volume confirmation + RSI not overbought
            if (close[i] > donchian_high_aligned[i] and 
                volume_confirm[i] and 
                rsi_not_overbought[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + volume confirmation + RSI not oversold
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_confirm[i] and 
                  rsi_not_oversold[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals