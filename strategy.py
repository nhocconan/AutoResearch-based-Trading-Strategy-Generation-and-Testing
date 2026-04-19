#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakouts with volume confirmation
# and RSI filter. Weekly Donchian provides strong trend structure, volume confirms
# breakout strength, and RSI prevents entries in overbought/oversold conditions.
# Designed to work in both bull and bear markets by requiring breakout alignment
# with weekly trend and filtering false signals with RSI.
# Target: 10-25 trades/year per symbol with disciplined entries.
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
    
    # Weekly Donchian channel (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    donchian_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > (volume_ma * 1.5)
    
    # RSI filter (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly Donchian high, with volume confirmation, RSI not overbought
            if (close[i] > donchian_high_aligned[i] and 
                volume_conf[i] and 
                rsi_values[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low, with volume confirmation, RSI not oversold
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_conf[i] and 
                  rsi_values[i] > 30):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly Donchian low or RSI overbought
            if (close[i] < donchian_low_aligned[i]) or (rsi_values[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly Donchian high or RSI oversold
            if (close[i] > donchian_high_aligned[i]) or (rsi_values[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals