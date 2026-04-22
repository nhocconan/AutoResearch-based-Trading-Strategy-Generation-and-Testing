#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Donchian breakouts capture momentum moves. Weekly pivot direction (based on prior week's range)
# filters for institutional bias. Volume spike (>2x 20-period average) confirms participation.
# Designed for low trade frequency (~15-30/year) to minimize fee decay while capturing strong moves.
# Works in both bull and bear markets by following higher timeframe weekly pivot direction.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot levels (based on prior week's range)
    # Weekly PP = (high + low + close) / 3
    # Weekly R1 = 2*PP - low
    # Weekly S1 = 2*PP - high
    pp_w = (high_w + low_w + close_w) / 3
    r1_w = 2 * pp_w - low_w
    s1_w = 2 * pp_w - high_w
    
    # Align weekly pivot to 6h timeframe (waits for weekly bar to close)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    
    # Calculate Donchian channels (20-period) on 6h close
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donch_high[i]
        lower = donch_low[i]
        r1_w_val = r1_w_aligned[i]
        s1_w_val = s1_w_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: break above Donchian high + weekly bullish bias + volume spike
            if price > upper and price > r1_w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + weekly bearish bias + volume spike
            elif price < lower and price < s1_w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low (reversal) or weekly bias turns bearish
                if price < lower or price < s1_w_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high (reversal) or weekly bias turns bullish
                if price > upper or price > r1_w_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0