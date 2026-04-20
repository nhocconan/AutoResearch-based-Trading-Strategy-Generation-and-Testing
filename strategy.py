#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d Trend Filter and Volume Confirmation
# Uses Donchian(20) channels for breakout signals with 1d EMA trend filter and volume confirmation.
# Enters long when price breaks above upper Donchian band with 1d uptrend and volume > 2x average.
# Enters short when price breaks below lower Donchian band with 1d downtrend and volume > 2x average.
# Exits when price returns to the opposite Donchian band or trend reverses.
# Donchian channels provide clear breakout levels, trend filter prevents counter-trend trades.
# Volume confirmation ensures institutional participation. Target: 80-160 total trades over 4 years (20-40/year).

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Donchian(20) channels on 4h data ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian channels: 20-period high/low
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values  # 30 * 4h = 5 days
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(40, n):  # Start after warmup
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_50_aligned[i]
        upper_band = high_roll[i]
        lower_band = low_roll[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(upper_band) or np.isnan(lower_band) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper Donchian band, 1d uptrend, volume confirmation
            if close_val > upper_band and close_val > ema_val and vol_ratio_val > 2.0:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short entry: price breaks below lower Donchian band, 1d downtrend, volume confirmation
            elif close_val < lower_band and close_val < ema_val and vol_ratio_val > 2.0:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: price returns to lower Donchian band or trend breaks
            if close_val < lower_band or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to upper Donchian band or trend breaks
            if close_val > upper_band or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals