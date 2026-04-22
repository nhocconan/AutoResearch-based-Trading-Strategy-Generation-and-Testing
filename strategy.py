#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low (EMA13 on close).
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) + price > 12h EMA50 (uptrend) + volume spike.
# Short when Bull Power < 0 and Bear Power > 0 (bearish momentum) + price < 12h EMA50 (downtrend) + volume spike.
# Uses 13-period EMA for Elder Ray (standard) and 50-period EMA on 12h for trend filter.
# Volume confirmation requires current volume > 1.8x 20-period average to filter noise.
# Designed to capture momentum shifts in both bull and bear markets by combining
# intraday momentum (Elder Ray) with higher timeframe trend (12h EMA).
# Targets 15-30 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h data
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate EMA13 for Elder Ray (on close)
    ema13 = pd.Series(prices['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = prices['high'].values - ema13  # High - EMA13
    bear_power = ema13 - prices['low'].values   # EMA13 - Low
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or 
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
        ema_val = ema_12h_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        
        # Volume filter: current volume > 1.8 * 20-period average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: bullish momentum + uptrend + volume spike
            if bp > 0 and br < 0 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish momentum + downtrend + volume spike
            elif bp < 0 and br > 0 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when momentum turns bearish or trend breaks
                if bp <= 0 or br >= 0 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when momentum turns bullish or trend breaks
                if bp >= 0 or br <= 0 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_12hEMA_Trend_Volume"
timeframe = "6h"
leverage = 1.0