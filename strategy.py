#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
# Uses weekly EMA200 to filter long-term trend (bullish above, bearish below) and daily Donchian
# breakouts for entry. Volume > 1.5x 20-day average confirms institutional participation.
# Designed for low trade frequency (~10-20/year) to minimize fee decay. Works in both bull
# and bear markets by following weekly trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for calculations (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 200-period EMA on 1d close for trend filter (proxy for weekly trend)
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Donchian channels (20-period high/low)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're already on 1d)
    # But we'll use the same pattern for consistency
    ema_200_aligned = ema_200_1d  # Already on 1d timeframe
    donchian_high_aligned = donchian_high
    donchian_low_aligned = donchian_low
    vol_ma_20_aligned = vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        ema_val = ema_200_aligned[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + above EMA200 + volume confirmation
            if price > upper_band and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + below EMA200 + volume confirmation
            elif price < lower_band and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below lower Donchian or trend breaks
                if price < lower_band or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above upper Donchian or trend breaks
                if price > upper_band or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_EMA200_Trend_Volume"
timeframe = "1d"
leverage = 1.0