#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot levels using previous day's HLC (no look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    pp_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pp_1d - prev_low_1d
    s1_1d = 2 * pp_1d - prev_high_1d
    r2_1d = pp_1d + (prev_high_1d - prev_low_1d)
    s2_1d = pp_1d - (prev_high_1d - prev_low_1d)
    r3_1d = pp_1d + 2 * (prev_high_1d - prev_low_1d)
    s3_1d = pp_1d - 2 * (prev_high_1d - prev_low_1d)
    r4_1d = pp_1d + 3 * (prev_high_1d - prev_low_1d)
    s4_1d = pp_1d - 3 * (prev_high_1d - prev_low_1d)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter (20-period average on 6h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily volatility filter (10-day average of absolute returns)
    daily_return = pd.Series(close_1d).pct_change().abs()
    vol_filter = pd.Series(daily_return).rolling(window=10, min_periods=10).mean().values
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        vol_filter_val = vol_filter_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        ema34 = ema34_aligned[i]
        
        # Volatility filter: only trade in normal volatility (avoid chop)
        vol_condition = vol_filter_val < 0.03  # Less than 3% daily volatility
        
        # Volume spike: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above R4 + volume spike + above EMA34 + normal vol
            if price > r4 and vol_spike and price > ema34 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + volume spike + below EMA34 + normal vol
            elif price < s4 and vol_spike and price < ema34 and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through S4/R4 or volatility increases
            exit_signal = False
            
            if position == 1:  # long
                if price < r4 or vol_filter_val > 0.05:  # High volatility exit
                    exit_signal = True
            elif position == -1:  # short
                if price > s4 or vol_filter_val > 0.05:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Pivot_R4_S4_Breakout_1dEMA34_Volume_Filter"
timeframe = "6h"
leverage = 1.0