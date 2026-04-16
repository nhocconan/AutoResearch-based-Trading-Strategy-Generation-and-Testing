#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with 1d RSI(14) momentum filter and volume confirmation.
# Long when price breaks above 1w Donchian(20) upper band AND 1d RSI > 50 AND 1d volume > 1.5x 20-period average.
# Short when price breaks below 1w Donchian(20) lower band AND 1d RSI < 50 AND 1d volume > 1.5x 20-period average.
# Exit when price crosses the 1w Donchian middle band (20-period average).
# Uses discrete position size 0.25. 1w filters provide signal direction, 1d provides entry timing and volume confirmation.
# Target: 50-100 total trades over 4 years (12-25/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data once before loop for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w Indicators: Donchian Channels (20) ===
    donchian_upper_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_middle_20_1w = (donchian_upper_20_1w + donchian_lower_20_1w) / 2.0
    
    # === 1d Indicators: RSI (14) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 0, rs)
    rsi_14_1d = 100 - (100 / (1 + rs))
    
    # === 1d Indicators: Volume MA (20) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20_1w)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle_20_1w)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        donchian_middle = donchian_middle_aligned[i]
        rsi = rsi_aligned[i]
        vol_ma = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            if price < donchian_middle:  # Exit when price crosses below middle band
                exit_signal = True
        
        elif position == -1:  # Short position
            if price > donchian_middle:  # Exit when price crosses above middle band
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND RSI > 50 AND volume > 1.5x 20-period avg
            if (price > donchian_upper) and (rsi > 50) and (vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND RSI < 50 AND volume > 1.5x 20-period avg
            elif (price < donchian_lower) and (rsi < 50) and (vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_1wDonchian20_1dRSI_VolumeConfirmation_V1"
timeframe = "1d"
leverage = 1.0