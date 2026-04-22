#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 AND volume spike.
# Short when Bear Power > 0 AND Bull Power < 0 AND price < 1d EMA50 AND volume spike.
# Exit when power signals reverse or volume drops below average.
# Works in bull (strong bull power) and bear (strong bear power) markets.
# Target: 15-30 trades/year to minimize fee drag on 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Elder Ray components (EMA13 of close)
    close = prices['close'].values
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    high = prices['high'].values
    low = prices['low'].values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA50 to 6h
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bp = bull_power[i]
        br = bear_power[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: Bull Power positive, Bear Power negative, price > EMA50, volume spike
            if bp > 0 and br < 0 and price > ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power positive, Bull Power negative, price < EMA50, volume spike
            elif br > 0 and bp < 0 and price < ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: power signals reverse or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Bull Power turns negative or Bear Power turns positive
                if bp <= 0 or br >= 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Bear Power turns negative or Bull Power turns positive
                if br <= 0 or bp >= 0:
                    exit_signal = True
            
            # Also exit if volume dries up significantly
            if vol < 0.6 * vol_ma:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_EMA50_Volume"
timeframe = "6h"
leverage = 1.0