#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Enters long when price breaks above 20-period high with 12h uptrend (close > EMA20) and volume > 1.8x average.
# Enters short when price breaks below 20-period low with 12h downtrend (close < EMA20) and volume > 1.8x average.
# Exits when price returns to 20-period EMA or trend reverses.
# Donchian captures breakouts, 12h EMA filters counter-trend trades, volume ensures institutional participation.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "4h_Donchian20_12hEMA20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === Donchian Channels (20-period high/low) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 20-period high and low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h EMA20 for trend filter ===
    twelve_h_close = df_12h['close'].values
    ema_20 = pd.Series(twelve_h_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  # 20-period average
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup
        # Get values
        close_val = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        ema_val = ema_20_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_20_val) or np.isnan(low_20_val) or np.isnan(ema_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 20-period high, 12h uptrend, volume confirmation
            if close_val > high_20_val and close_val > ema_val and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short entry: price breaks below 20-period low, 12h downtrend, volume confirmation
            elif close_val < low_20_val and close_val < ema_val and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: price returns to 20-period EMA or trend breaks
            if close_val <= ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 20-period EMA or trend breaks
            if close_val >= ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals