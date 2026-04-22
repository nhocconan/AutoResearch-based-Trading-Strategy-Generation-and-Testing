#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA40 trend + volume confirmation
# Long when price breaks above upper Donchian(20) and close > 1w EMA40 with volume spike
# Short when price breaks below lower Donchian(20) and close < 1w EMA40 with volume spike
# Exit when price crosses back through the opposite Donchian band
# Uses weekly EMA40 to filter trend direction, reducing false breakouts
# Designed for low trade frequency (~10-25/year) with edge in trending markets
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)

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
    # Upper band: highest high over last 20 periods
    # Lower band: lowest low over last 20 periods
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Load 1w data for EMA40 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(ema_40_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        close_1d_val = close_1d[i]
        ema_40_val = ema_40_1w_aligned[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, close > weekly EMA40, volume spike
            if price > upper_band and close_1d_val > ema_40_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian, close < weekly EMA40, volume spike
            elif price < lower_band and close_1d_val < ema_40_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through opposite Donchian band
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below lower Donchian band
                if price < lower_band:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above upper Donchian band
                if price > upper_band:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA40_Volume"
timeframe = "1d"
leverage = 1.0