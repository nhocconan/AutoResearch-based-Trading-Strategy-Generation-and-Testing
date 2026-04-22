#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (bull/bear power) with 1-week EMA trend filter and volume confirmation.
# Elder Ray measures bull power (high - EMA) and bear power (low - EMA) to assess trend strength.
# Only take long when bull power > 0 and bear power rising; short when bear power < 0 and bull power falling.
# 1-week EMA filter ensures alignment with higher timeframe trend.
# Volume confirmation filters out low-conviction moves.
# Designed to work in both bull and bear markets by combining trend (EMA) and momentum (Elder Ray).
# Targets 20-35 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1-week data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 21-period EMA on 1w data
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 13-period EMA for Elder Ray (using high/low/close)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    ema_close = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_close  # High minus EMA
    bear_power = low - ema_close   # Low minus EMA
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or 
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
        ema_val = ema_1w_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        # Volume filter: current volume > 1.6 * 20-period average
        vol_spike = vol > 1.6 * vol_ma
        
        if position == 0:
            # Long conditions: bull power positive, bear power rising, above weekly EMA, volume spike
            if bull > 0 and bear > bear_power[i-1] and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bear power negative, bull power falling, below weekly EMA, volume spike
            elif bear < 0 and bull < bull_power[i-1] and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bull power turns negative or bear power rises sharply
                if bull <= 0 or bear > 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bear power turns positive or bull power rises sharply
                if bear >= 0 or bull > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1wEMA_Trend_Volume"
timeframe = "6h"
leverage = 1.0