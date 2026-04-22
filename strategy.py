#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume spike and 1w EMA20 trend filter.
# Long when price breaks above 20-day high + volume spike + price > 1w EMA20
# Short when price breaks below 20-day low + volume spike + price < 1w EMA20
# Exit when price crosses back through 10-day EMA or volume drops below 80% of average.
# Uses weekly trend filter to avoid counter-trend trades in bear markets.
# Target: 10-25 trades/year to stay under fee drag limits.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    # Upper = max(high over last 20 days)
    # Lower = min(low over last 20 days)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donch_up = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Load 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 1d
    donch_up_aligned = align_htf_to_ltf(prices, df_1d, donch_up)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1d, ema20_1w)
    
    # 10-day EMA for exit
    ema10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_aligned = align_htf_to_ltf(prices, df_1d, ema10_1d)
    
    # Volume spike filter (20-period average)
    volume = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donch_up_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(ema10_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume[i]
        vol_ma = vol_ma_aligned[i]
        upper = donch_up_aligned[i]
        lower = donch_low_aligned[i]
        ema20w = ema20_1w_aligned[i]
        ema10 = ema10_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + volume spike + price > weekly EMA20
            if price > upper and vol_spike and price > ema20w:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + volume spike + price < weekly EMA20
            elif price < lower and vol_spike and price < ema20w:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through 10-day EMA or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below 10-day EMA or volume dries up
                if price < ema10 or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above 10-day EMA or volume dries up
                if price > ema10 or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA20_Volume"
timeframe = "1d"
leverage = 1.0