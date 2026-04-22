#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1-week EMA13 trend filter and volume spike confirmation.
# Elder Ray uses Bull Power (High - EMA13) and Bear Power (Low - EMA13) to measure bull/bear strength.
# When Bull Power > 0 and Bear Power < 0 with divergence, it indicates strong trend.
# Combined with 1-week EMA13 trend filter and volume spikes (>2x 20-period average),
# this captures institutional moves while avoiding chop. Designed for low trade frequency (~15-30/year)
# to minimize fee decay. Works in both bull and bear markets by following higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1-week data for EMA13 calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 13-period EMA on 1w close for trend filter
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1-week EMA to 6h timeframe (waits for 1w bar to close)
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Calculate 13-period EMA on 6h close for Elder Ray
    close_6h = prices['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13_6h  # High - EMA13
    bear_power = low_6h - ema_13_6h   # Low - EMA13
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1w_aligned[i]) or 
            np.isnan(ema_13_6h[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_1w_val = ema_13_1w_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        # Elder Ray conditions: strong bull/bear power with alignment
        # Strong bull: Bull Power > 0 and Bear Power < 0 (both conditions)
        # Strong bear: Bear Power < 0 and Bull Power > 0 (both conditions) - actually same as above
        # We need: Bull Power > 0 AND Bear Power < 0 for bullish conviction
        #          Bull Power < 0 AND Bear Power > 0 for bearish conviction
        bullish_elder = bull_val > 0 and bear_val < 0
        bearish_elder = bull_val < 0 and bear_val > 0
        
        if position == 0:
            # Long conditions: bullish Elder Ray + price above 1w EMA + volume spike
            if bullish_elder and price > ema_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish Elder Ray + price below 1w EMA + volume spike
            elif bearish_elder and price < ema_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Elder Ray turns bearish or price breaks below 1w EMA
                if not bullish_elder or price < ema_1w_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Elder Ray turns bullish or price breaks above 1w EMA
                if not bearish_elder or price > ema_1w_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1wEMA13_Volume"
timeframe = "6h"
leverage = 1.0