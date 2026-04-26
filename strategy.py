#!/usr/bin/env python3
"""
6h_ADX_Alligator_Trend_v1
Hypothesis: Combine ADX trend strength with Williams Alligator crossover for high-probability 6h trend entries.
ADX > 25 filters for trending markets; Alligator (jaw/teeth/lips) cross confirms direction.
Uses 12h HTF trend filter (EMA50) to avoid counter-trend trades. Volume spike confirms breakout quality.
Designed for low turnover (~80-120 trades over 4 years) to minimize fee drag while capturing strong trends.
Works in both bull/bear markets by requiring trend alignment and avoiding chop via ADX threshold.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA(50) for HTF trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams Alligator on 6h: jaw=13, teeth=8, lips=5 (all SMMA)
    def smma(src, length):
        result = np.full_like(src, np.nan, dtype=float)
        if len(src) < length:
            return result
        result[length-1] = np.mean(src[:length])
        for i in range(length, len(src)):
            result[i] = (result[i-1] * (length-1) + src[i]) / length
        return result
    
    jaw = smma(high, 13)  # Alligator jaw (high)
    teeth = smma(low, 8)   # Alligator teeth (low)
    lips = smma(close, 5)  # Alligator lips (close)
    
    # ADX(14) for trend strength
    def calculate_adx(high, low, close, length=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            move_up = high[i] - high[i-1]
            move_down = low[i-1] - low[i]
            
            plus_dm[i] = move_up if move_up > move_down and move_up > 0 else 0
            minus_dm[i] = move_down if move_down > move_up and move_down > 0 else 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Handle first bar
        tr[0] = high[0] - low[0]
        
        # Smoothed values
        atr = np.zeros_like(high)
        atr[:length] = np.nan
        if len(high) >= length:
            atr[length-1] = np.mean(tr[:length])
            for i in range(length, len(high)):
                atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        if len(high) >= length:
            plus_sm = np.zeros_like(high)
            minus_sm = np.zeros_like(high)
            plus_sm[length-1] = np.sum(plus_dm[:length])
            minus_sm[length-1] = np.sum(minus_dm[:length])
            for i in range(length, len(high)):
                plus_sm[i] = plus_sm[i-1] - (plus_sm[i-1] / length) + plus_dm[i]
                minus_sm[i] = minus_sm[i-1] - (minus_sm[i-1] / length) + minus_dm[i]
            
            plus_di = (plus_sm / atr) * 100
            minus_di = (minus_sm / atr) * 100
            dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            dx[plus_di + minus_di == 0] = 0
        
        adx = np.zeros_like(high)
        adx[:length] = np.nan
        if len(high) >= 2*length-1:
            adx[2*length-2] = np.mean(dx[length-1:2*length-1])
            for i in range(2*length-1, len(high)):
                adx[i] = (adx[i-1] * (length-1) + dx[i]) / length
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Align HTF indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume confirmation: 1.8x median volume (balanced for frequency)
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 12h EMA (50), ADX (2*14-1=27), jaw (13), teeth (8), lips (5), volume (30)
    start_idx = max(50, 27, 13, 8, 5, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_12h_val = ema_50_12h_aligned[i]
        adx_val = adx_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        if position == 0:
            # Long: ADX > 25 (trending), lips > teeth > jaw (bullish alignment), volume spike, HTF uptrend
            long_signal = (adx_val > 25) and \
                          (lips_val > teeth_val) and \
                          (teeth_val > jaw_val) and \
                          (volume_val > 1.8 * vol_median_val) and \
                          (close_val > ema_50_12h_val)
            # Short: ADX > 25 (trending), lips < teeth < jaw (bearish alignment), volume spike, HTF downtrend
            short_signal = (adx_val > 25) and \
                           (lips_val < teeth_val) and \
                           (teeth_val < jaw_val) and \
                           (volume_val > 1.8 * vol_median_val) and \
                           (close_val < ema_50_12h_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                # Calculate ATR-based stop (using 14-period ATR approximation)
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                atr_stop = close_val - 2.5 * tr  # 2.5x ATR stop
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                # Calculate ATR-based stop (using 14-period ATR approximation)
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                atr_stop = close_val + 2.5 * tr  # 2.5x ATR stop
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr_stop = max(atr_stop, high[i] - 2.5 * tr)
            # Exit: trailing stop hit or Alligator reversal (lips < teeth) after minimum holding period
            if bars_since_entry >= 6 and ((low[i] < atr_stop) or (lips_val < teeth_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr_stop = min(atr_stop, low[i] + 2.5 * tr)
            # Exit: trailing stop hit or Alligator reversal (lips > teeth) after minimum holding period
            if bars_since_entry >= 6 and ((high[i] > atr_stop) or (lips_val > teeth_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_Trend_v1"
timeframe = "6h"
leverage = 1.0