#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Control_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly: Calculate SMA200 for trend filter ===
    weekly_close = df_1w['close'].values
    weekly_sma200 = pd.Series(weekly_close).rolling(window=200, min_periods=200).mean().values
    weekly_sma200_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma200)
    
    # === Daily: Calculate daily close and volume ===
    daily_close = prices['close'].values
    daily_volume = prices['volume'].values
    
    # Daily volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = daily_volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Daily: Calculate Camarilla pivot levels (using previous day's data) ===
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Use previous day's OHLC for today's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Set first day's values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels: R1, S1
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after SMA200 warmup
        # Get values
        close_val = daily_close[i]
        r1_level = camarilla_r1[i]
        s1_level = camarilla_s1[i]
        vol_ratio_val = vol_ratio[i]
        weekly_trend = weekly_sma200_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_level) or np.isnan(s1_level) or np.isnan(vol_ratio_val) or 
            np.isnan(weekly_trend)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and weekly uptrend
            if (close_val > r1_level and   # Break above R1
                vol_ratio_val > 2.0 and    # Strong volume confirmation
                close_val > weekly_trend): # Weekly uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and weekly downtrend
            elif (close_val < s1_level and   # Break below S1
                  vol_ratio_val > 2.0 and    # Strong volume confirmation
                  close_val < weekly_trend): # Weekly downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below R1 (reversion to mean)
            if close_val < r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above S1 (reversion to mean)
            if close_val > s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals