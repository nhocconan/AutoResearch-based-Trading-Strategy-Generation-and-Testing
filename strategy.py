#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI(14) extreme + 1d Donchian(20) breakout with volume confirmation.
# Long when 4h RSI < 30 (oversold) AND price breaks above 1d Donchian upper band AND volume > 1.5x 20-period average.
# Short when 4h RSI > 70 (overbought) AND price breaks below 1d Donchian lower band AND volume > 1.5x 20-period average.
# Uses 1h for entry timing precision, 4h/1d for signal direction to minimize trades.
# Session filter: 08-20 UTC to avoid low-volume periods.
# Discrete position size 0.20 to control risk and fee drag.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag death spiral.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for RSI(14)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Get 1d data once before loop for Donchian(20)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h Indicators: RSI(14) ===
    delta = pd.Series(close_4h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_values = rsi_4h.values
    
    # === 1d Indicators: Donchian(20) ===
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align all indicators to primary timeframe (1h)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_values)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # === 1h Indicators: Volume filter (20-period average) ===
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian20 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours for efficiency
    hours = prices.index.hour
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        rsi = rsi_4h_aligned[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when RSI > 50 (momentum fading) OR price < lower band (breakdown)
            if (rsi > 50) or (price < lower_band):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when RSI < 50 (momentum fading) OR price > upper band (breakout)
            if (rsi < 50) or (price > upper_band):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_ok = vol > 1.5 * vol_ma
            
            # LONG: 4h RSI < 30 (oversold) AND price breaks above upper band AND volume OK
            if (rsi < 30) and (price > upper_band) and volume_ok:
                signals[i] = 0.20
                position = 1
            
            # SHORT: 4h RSI > 70 (overbought) AND price breaks below lower band AND volume OK
            elif (rsi > 70) and (price < lower_band) and volume_ok:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20  # maintain position
    
    return signals

name = "1h_4hRSI14_Extreme_1dDonchian20_VolumeFilter_V1"
timeframe = "1h"
leverage = 1.0