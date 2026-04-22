#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 200-day EMA trend filter + weekly Donchian(10) breakout + volume confirmation.
# Uses weekly Donchian channels for breakout signals in the direction of the daily 200-EMA trend.
# In uptrend (price > daily EMA200): long on breakout above weekly Donchian high with volume spike.
# In downtrend (price < daily EMA200): short on breakout below weekly Donchian low with volume spike.
# Designed to capture major trends while avoiding counter-trend whipsaws.
# Targets 10-25 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data for EMA200 (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200-day EMA
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Load weekly data for Donchian channels (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 10-period weekly Donchian channels
    donch_high_1w = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    donch_low_1w = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    
    # Align indicators to daily timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema200_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema200_val = ema200_aligned[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Determine trend based on daily EMA200
            if price > ema200_val:  # Uptrend
                if price > upper and vol_spike:
                    signals[i] = 0.25
                    position = 1
            elif price < ema200_val:  # Downtrend
                if price < lower and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: opposite Donchian break or loss of trend
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on break below weekly Donchian low or price below EMA200
                if price < lower or price < ema200_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on break above weekly Donchian high or price above EMA200
                if price > upper or price > ema200_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_EMA200_WeeklyDonchian10_Trend"
timeframe = "1d"
leverage = 1.0