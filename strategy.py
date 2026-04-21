#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly High-Low Range for volatility regime ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly range (high - low)
    weekly_range = high_1w - low_1w
    
    # Weekly range percentile (52-week lookback)
    range_percentile = pd.Series(weekly_range).rolling(window=52, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align range percentile to daily timeframe
    range_percentile_aligned = align_htf_to_ltf(prices, df_1w, range_percentile)
    
    # === Daily price position within weekly range ===
    # Use daily close relative to weekly range
    weekly_low = df_1w['low'].values
    weekly_high = df_1w['high'].values
    
    # Align weekly high/low to daily
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    
    # Calculate position: 0 = at weekly low, 1 = at weekly high
    range_width = weekly_high_aligned - weekly_low_aligned
    range_width = np.where(range_width == 0, 1, range_width)  # avoid division by zero
    price_position = (prices['close'].values - weekly_low_aligned) / range_width
    
    # === Daily volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(range_percentile_aligned[i]) or 
            np.isnan(price_position[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        range_pct = range_percentile_aligned[i]
        pos_in_range = price_position[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long in low volatility (range) + lower third of weekly range + volume
            if (range_pct < 30 and  # Low volatility regime
                pos_in_range < 0.33 and  # Lower third of weekly range
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short in low volatility (range) + upper third of weekly range + volume
            elif (range_pct < 30 and   # Low volatility regime
                  pos_in_range > 0.66 and  # Upper third of weekly range
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when volatility increases (trending regime) or moves to middle third
            if position == 1 and (range_pct > 70 or pos_in_range > 0.66):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (range_pct > 70 or pos_in_range < 0.33):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyRange_Volatility_Position_Volume"
timeframe = "1d"
leverage = 1.0