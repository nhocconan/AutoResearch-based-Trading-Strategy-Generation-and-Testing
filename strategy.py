#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA trend filter + volume spike
# Long when price > Alligator's Jaw (green line) + price > 1d EMA34 + volume > 2x 20-period avg
# Short when price < Alligator's Jaw + price < 1d EMA34 + volume > 2x 20-period avg
# Exit when price crosses back below/above Jaw or trend reverses
# Williams Alligator uses smoothed medians (3-period SMAs) to avoid whipsaw in chop
# Designed for low trade frequency (~15-30/year) to minimize fee drain. Works in bull/bear by
# combining trend-following with trend filter and volume confirmation to avoid false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (13,8,5 periods smoothed with 8,5,3 periods)
    # Jaw (blue): 13-period SMMA smoothed by 8 periods
    # Teeth (red): 8-period SMMA smoothed by 5 periods
    # Lips (green): 5-period SMMA smoothed by 3 periods
    # We use the Lips (green, fastest) as trigger line
    close = prices['close'].values
    
    # Smoothed Moving Average (SMMA) - same as Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate Alligator components
    # Lips: 5-period SMMA of median price, smoothed by 3 periods
    median_price = (high_1d + low_1d) / 2.0
    lips_raw = smma(median_price, 5)
    lips = smma(lips_raw, 3)
    
    # Teeth: 8-period SMMA of median price, smoothed by 5 periods
    teeth_raw = smma(median_price, 8)
    teeth = smma(teeth_raw, 5)
    
    # Jaw: 13-period SMMA of median price, smoothed by 8 periods
    jaw_raw = smma(median_price, 13)
    jaw = smma(jaw_raw, 8)
    
    # Align Alligator lines to 12h timeframe
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(lips_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        lips_val = lips_aligned[i]
        jaw_val = jaw_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price > Lips (Alligator's green line) + uptrend + volume spike
            if price > lips_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Lips + downtrend + volume spike
            elif price < lips_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses Lips or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below Lips or trend turns down
                if price <= lips_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above Lips or trend turns up
                if price >= lips_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0