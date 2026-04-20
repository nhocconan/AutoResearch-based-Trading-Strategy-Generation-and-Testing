#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_Camarilla_R1S1_Breakout_Volume_Control"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for warmup
        return np.zeros(n)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least 20 periods for calculations
        return np.zeros(n)
    
    # === 4h: Calculate Camarilla pivot levels (using previous day's data) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Use previous day's OHLC for today's levels
    prev_close = np.roll(close_4h, 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    
    # Set first day's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels: R1, S1
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align 4h indicators to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # === 1h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_level) or np.isnan(s1_level) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation
            if (close_val > r1_level and   # Break above R1
                vol_ratio_val > 2.0):      # Strong volume confirmation
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 with volume confirmation
            elif (close_val < s1_level and   # Break below S1
                  vol_ratio_val > 2.0):      # Strong volume confirmation
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below R1 (reversion to mean)
            if close_val < r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price rises back above S1 (reversion to mean)
            if close_val > s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals