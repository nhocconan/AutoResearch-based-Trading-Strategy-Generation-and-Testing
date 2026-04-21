#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R + 1d EMA trend + volume spike for mean-reversion entries.
Williams %R identifies overbought/oversold conditions on 4h, filtered by 1d EMA trend
and confirmed by volume spikes. Works in both bull and bear markets by fading extremes
in the direction of the higher timeframe trend. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h Williams %R (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    highest_high = np.maximum.accumulate(high_4h)
    lowest_low = np.minimum.accumulate(low_4h)
    
    # For Williams %R, we need the highest high and lowest low over the lookback period
    williams_r = np.full_like(close_4h, -50.0)  # Default to neutral
    for i in range(13, len(close_4h)):
        period_high = np.max(high_4h[i-13:i+1])
        period_low = np.min(low_4h[i-13:i+1])
        if period_high != period_low:
            williams_r[i] = -100 * (period_high - close_4h[i]) / (period_high - period_low)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Williams %R signals: oversold (< -80) for long, overbought (> -20) for short
    williams_oversold = (williams_r < -80).astype(float)
    williams_overbought = (williams_r > -20).astype(float)
    
    # Align to 4h timeframe
    williams_oversold_4h = align_htf_to_ltf(prices, df_4h, williams_oversold)
    williams_overbought_4h = align_htf_to_ltf(prices, df_4h, williams_overbought)
    
    # Align 1d EMA to 4h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 4h volume / 20-period average
    vol_ma_20 = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = df_4h['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(williams_oversold_4h[i]) or np.isnan(williams_overbought_4h[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R values for current bar
        oversold_signal = williams_oversold_4h[i] > 0.5
        overbought_signal = williams_overbought_4h[i] > 0.5
        
        vol_ratio_val = vol_ratio_aligned[i]
        vol_threshold = 2.0  # Volume spike filter
        
        ema_trend = ema_34_1d_aligned[i]
        # Simple trend: price above EMA = uptrend, below = downtrend
        # We'll use the 4h close price to determine trend alignment
        # Get current 4h close price by finding the corresponding 4h bar
        # Since we're aligned, we can use the price at the 4h boundary
        # For simplicity, we'll use current price vs EMA
        price_close = prices['close'].iloc[i]
        
        if position == 0:
            # Enter long: Williams %R oversold + price above 1d EMA (uptrend) + volume spike
            if (oversold_signal and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought + price below 1d EMA (downtrend) + volume spike
            elif (overbought_signal and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R returns to neutral range (-50) or opposite extreme
            # Check if Williams %R has crossed back above -50 (for long) or below -50 (for short)
            # We need the actual Williams %R value, not just the signal
            williams_current = williams_r[-(len(prices)-i)] if i < len(williams_r) else williams_r[-1]
            # Actually, let's use the aligned Williams %R values we have
            # We'll reconstruct the Williams %R alignment for the actual values
            williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
            williams_val = williams_r_aligned[i]
            
            if position == 1 and (williams_val > -50 or williams_val > -20):  # Exited oversold or reached overbought
                signals[i] = 0.0
                position = 0
            elif position == -1 and (williams_val < -50 or williams_val < -80):  # Exited overbought or reached oversold
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_Volume_Spike"
timeframe = "4h"
leverage = 1.0