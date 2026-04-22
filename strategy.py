# 6h Donchian breakout with weekly pivot direction and volume confirmation
# Long: price breaks above 6h Donchian high (20) + price > weekly pivot + volume spike
# Short: price breaks below 6h Donchian low (20) + price < weekly pivot + volume spike
# Exit: price crosses back through Donchian midpoint or volume drops
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets
# Target: 15-35 trades/year to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot point (standard)
    # PP = (H+L+C)/3
    pp_w = (high_w + low_w + close_w) / 3
    
    # Load daily data for volume context (optional filter)
    df_d = get_htf_data(prices, '1d')
    volume_d = df_d['volume'].values
    
    # Calculate Donchian channels (20-period) on 6h data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Upper band: highest high of last 20 periods
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    # Middle line: average of upper and lower
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume spike filter (20-period average on 6h data)
    volume_6h = prices['volume'].values
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly pivot to 6h
    pp_w_aligned = align_htf_to_ltf(prices, df_w, pp_w)
    
    # Align daily volume average (for context)
    vol_ma_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    vol_ma_d_aligned = align_htf_to_ltf(prices, df_d, vol_ma_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(pp_w_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(vol_ma_d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ma = vol_ma_20[i]
        vol_ma_d = vol_ma_d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        pp = pp_w_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average AND above daily average
        vol_spike = vol > 1.5 * vol_ma and vol > vol_ma_d
        
        if position == 0:
            # Long conditions: break above Donchian high + above weekly pivot + volume spike
            if price > upper and price > pp and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + below weekly pivot + volume spike
            elif price < lower and price < pp and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through Donchian midpoint or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below midpoint or volume drops significantly
                if price < mid or vol < 0.6 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above midpoint or volume drops significantly
                if price > mid or vol < 0.6 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0