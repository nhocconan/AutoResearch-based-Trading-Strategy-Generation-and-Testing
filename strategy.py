#!/usr/bin/env python3
"""
12h Williams Alligator with Volume Spike and 1d Trend Filter
Williams Alligator identifies convergence/divergence of smoothed moving averages (Jaw, Teeth, Lips).
Convergence (all lines intertwined) indicates ranging/choppy market - trade reversals at extremes.
Divergence (lines separated and ordered) indicates trending market - trade in direction of alignment.
Volume spike confirms institutional participation. Designed for 15-25 trades/year to minimize fee drag.
Works in bull markets (buy on bullish alignment + pullback) and bear markets (sell on bearish alignment + rally).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    # First value is simple SMA
    result[period-1] = np.mean(arr[0:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
    for i in range(period, n):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def williams_alligator(high, low, close):
    """Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3)"""
    # Typical price
    typical = (high + low + close) / 3.0
    jaw = smma(typical, 13)  # Blue line
    teeth = smma(typical, 8)  # Red line
    lips = smma(typical, 5)   # Green line
    # Shift as per Williams: Jaw 8 bars, Teeth 5 bars, Lips 3 bars
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First values become NaN after roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                  np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values (Wilder smoothing)
        atr = np.full(n, np.nan)
        plus_dm_smooth = np.full(n, np.nan)
        minus_dm_smooth = np.full(n, np.nan)
        
        # Initial values
        if n >= period:
            atr[period-1] = np.nanmean(tr[1:period+1])
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period+1])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period+1])
            
            # Wilder smoothing
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX
        adx = np.full(n, np.nan)
        if n >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams Alligator on 12h data
    jaw, teeth, lips = williams_alligator(high, low, close)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need Alligator calculation and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_12h[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX threshold
        trending = adx_1d_12h[i] >= 25
        ranging = adx_1d_12h[i] < 25
        
        # Alligator relationships
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Convergence: all lines close together (ranging market)
        # Divergence: lines separated and ordered (trending market)
        converged = (abs(jaw_val - teeth_val) < 0.001 * close[i] and 
                    abs(teeth_val - lips_val) < 0.001 * close[i] and
                    abs(lips_val - jaw_val) < 0.001 * close[i])
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_aligned = lips_val > teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_aligned = lips_val < teeth_val < jaw_val
        
        if position == 0:
            if ranging and converged:
                # Ranging market: trade reversals at extremes
                # Look for price extreme near/alligator lines
                if (low[i] <= lips_val * 1.002 and vol_spike[i]):  # Near lips (support)
                    signals[i] = 0.25
                    position = 1
                elif (high[i] >= lips_val * 0.998 and vol_spike[i]):  # Near lips (resistance)
                    signals[i] = -0.25
                    position = -1
            elif trending:
                # Trending market: trade with alignment on pullbacks
                if bullish_aligned and vol_spike[i]:
                    # Buy on pullback to teeth or jaw in uptrend
                    if close[i] <= teeth_val * 1.005:  # Near teeth
                        signals[i] = 0.25
                        position = 1
                elif bearish_aligned and vol_spike[i]:
                    # Sell on rally to teeth or jaw in downtrend
                    if close[i] >= teeth_val * 0.995:  # Near teeth
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Long exit: convergence (ranging) or bearish alignment
            if converged or bearish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: convergence (ranging) or bullish alignment
            if converged or bullish_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ADX_Volume"
timeframe = "12h"
leverage = 1.0