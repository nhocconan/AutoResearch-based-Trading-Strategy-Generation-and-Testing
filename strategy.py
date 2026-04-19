#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper channel (20-period high), price > 1d EMA50, and volume > 1.5x 4h average volume.
# Short when price breaks below 4h Donchian lower channel (20-period low), price < 1d EMA50, and volume > 1.5x 4h average volume.
# Exit when price crosses the 4h Donchian opposite channel (long exits at lower channel, short exits at upper channel).
# Uses Donchian for breakout structure, EMA for trend filter, volume for confirmation.
# Target: 20-40 trades/year per symbol to stay within frequency limits.
name = "4h_Donchian20_EMA50_Volume_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA50 on daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h Donchian channels (20-period high/low)
    donchian_window = 20
    dc_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Get 4h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = donchian_window  # Ensure Donchian channels are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(dc_high[i]) or 
            np.isnan(dc_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        dc_high_val = dc_high[i]
        dc_low_val = dc_low[i]
        ema_50 = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above Donchian upper channel, above EMA50, with volume confirmation
            if price > dc_high_val and price > ema_50 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower channel, below EMA50, with volume confirmation
            elif price < dc_low_val and price < ema_50 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian lower channel
            if price < dc_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian upper channel
            if price > dc_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals