#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d EMA34 trend filter and volume confirmation.
# Elder Ray = Bull Power (high - EMA13) and Bear Power (low - EMA13).
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, in uptrend (1d EMA34 rising).
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, in downtrend (1d EMA34 falling).
# Volume > 1.3x 20-period average confirms strength.
# Works in bull/bear: EMA34 trend filter ensures alignment with higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False).values
    
    # Align EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 13-period EMA for Elder Ray (using 6h data)
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).values
    
    # Calculate Elder Ray components
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema_13  # High minus EMA13
    bear_power = low - ema_13   # Low minus EMA13
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(ema_13[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        bp = bull_power[i]
        bp_prev = bull_power[i-1] if i > 0 else bp
        bp_change = bp - bp_prev
        
        br = bear_power[i]
        br_prev = bear_power[i-1] if i > 0 else br
        br_change = br - br_prev
        
        volume = prices['volume'].iloc[i]
        ema_34_val = ema_34_aligned[i]
        ema_34_prev = ema_34_aligned[i-1] if i > 0 else ema_34_val
        ema_34_slope = ema_34_val - ema_34_prev
        
        # Volume confirmation
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long conditions: Bull Power positive and rising, Bear Power negative, Uptrend
            if bp > 0 and bp_change > 0 and br < 0 and ema_34_slope > 0 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power negative and falling, Bull Power positive, Downtrend
            elif br < 0 and br_change < 0 and bp > 0 and ema_34_slope < 0 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Bull Power turns negative or trend weakens
                if bp <= 0 or ema_34_slope <= 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Bear Power turns positive or trend weakens
                if br >= 0 or ema_34_slope >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0