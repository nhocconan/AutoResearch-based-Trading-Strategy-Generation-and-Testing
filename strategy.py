#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# Long when price breaks above Donchian upper band and price > weekly EMA50
# Short when price breaks below Donchian lower band and price < weekly EMA50
# Exit when price crosses back through Donchian midpoint
# Uses volume spike (>1.5x 20-day average) for confirmation
# Designed for low trade frequency (~10-25/year) with strong trend-following edge
# Works in bull markets (breakouts above EMA50) and bear markets (breakdowns below EMA50)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period)
    # Upper band: highest high over last 20 periods
    # Lower band: lowest low over last 20 periods
    # Middle band: average of upper and lower
    lookback = 20
    upper_band = np.full(len(high_1d), np.nan)
    lower_band = np.full(len(low_1d), np.nan)
    
    for i in range(lookback - 1, len(high_1d)):
        upper_band[i] = np.max(high_1d[i - lookback + 1:i + 1])
        lower_band[i] = np.min(low_1d[i - lookback + 1:i + 1])
    
    middle_band = (upper_band + lower_band) / 2.0
    
    # Calculate weekly EMA50
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    middle_band_aligned = align_htf_to_ltf(prices, df_1d, middle_band)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume spike filter (20-day average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or 
            np.isnan(middle_band_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        middle = middle_band_aligned[i]
        ema50 = ema_50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper band, price > EMA50, volume spike
            if price > upper and price > ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band, price < EMA50, volume spike
            elif price < lower and price < ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through middle band
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below middle band
                if price < middle:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above middle band
                if price > middle:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0