#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above Donchian(20) high + price > weekly pivot (bullish bias) + volume spike
# Short when price breaks below Donchian(20) low + price < weekly pivot (bearish bias) + volume spike
# Exit when price crosses Donchian midpoint or weekly pivot flips
# Weekly pivot provides structural bias from higher timeframe, reducing false breakouts
# Volume spike confirms institutional participation
# Designed for low trade frequency (~15-35/year) to minimize fee drain in bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (done once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Calculate Donchian channels on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 20-period Donchian high and low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Calculate volume spike filter
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(pp_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        dh = donch_high[i]
        dl = donch_low[i]
        dm = donch_mid[i]
        pp = pp_1w_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-period average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: break above Donchian high + above weekly pivot + volume spike
            if price > dh and price > pp and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + below weekly pivot + volume spike
            elif price < dl and price < pp and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses Donchian midpoint or weekly pivot flips
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price falls below Donchian midpoint or weekly pivot
                if price < dm or price < pp:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price rises above Donchian midpoint or weekly pivot
                if price > dm or price > pp:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0