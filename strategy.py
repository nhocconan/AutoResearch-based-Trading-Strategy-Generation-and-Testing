#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1-week EMA13 trend filter and volume confirmation.
# Elder Ray measures bull power (high - EMA) and bear power (low - EMA) to identify trend strength.
# In strong trends, bull/bear power expands with price; in reversals, power diminishes before price.
# Combined with 1-week EMA13 for primary trend direction and volume spikes (>2x 20-period average),
# this captures sustained moves while avoiding false breakouts. Designed for low trade frequency
# (~20-35/year) to minimize fee decay. Works in both bull and bear markets by following
# higher timeframe trend and confirming with institutional participation (volume).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data for EMA13 calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA on daily close for trend filter
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align daily EMA to 6h timeframe (waits for daily bar to close)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 13-period EMA on 6h close for Elder Ray
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_13_aligned[i]) or 
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
        ema_13_val = ema_13_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        # Elder Ray conditions: expanding power in direction of trend
        # Bullish: bull power increasing AND above zero (buyers in control)
        # Bearish: bear power decreasing AND below zero (sellers in control)
        bullish_ray = bull_val > 0 and bull_val > bull_power[i-1]
        bearish_ray = bear_val < 0 and bear_val < bear_power[i-1]
        
        if position == 0:
            # Long conditions: bullish Elder Ray + price above EMA13 + volume spike
            if bullish_ray and price > ema_13_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish Elder Ray + price below EMA13 + volume spike
            elif bearish_ray and price < ema_13_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bull power turns negative or weakens or price breaks below EMA
                if bull_val <= 0 or bull_val < bull_power[i-1] or price < ema_13_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bear power turns positive or weakens or price breaks above EMA
                if bear_val >= 0 or bear_val > bear_power[i-1] or price > ema_13_val:
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