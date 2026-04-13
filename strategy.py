#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 12h ADX trend filter + volume confirmation
    # Long: price > Camarilla R4 + 12h ADX > 25 + volume > 1.5x 20-period average
    # Short: price < Camarilla S4 + 12h ADX > 25 + volume > 1.5x 20-period average
    # Exit: price crosses Camarilla H3/L3 levels OR ADX < 20 (range regime)
    # Uses 12h Camarilla pivots calculated from prior 12h bar's OHLC for no look-ahead
    # ADX filter ensures we only trade in trending markets, reducing whipsaw in chop
    # Volume confirmation adds conviction to breakouts
    # Target: 12-37 trades/year (50-150 over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for each 12h bar
    # Based on prior 12h bar's OHLC (no look-ahead)
    camarilla_h3 = np.full(len(close_12h), np.nan)
    camarilla_l3 = np.full(len(close_12h), np.nan)
    camarilla_h4 = np.full(len(close_12h), np.nan)
    camarilla_l4 = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):  # Start from 1 to use prior bar
        # Prior 12h bar's OHLC
        phigh = high_12h[i-1]
        plow = low_12h[i-1]
        pclose = close_12h[i-1]
        
        pivot = (phigh + plow + pclose) / 3
        range_val = phigh - plow
        
        camarilla_h3[i] = pclose + range_val * 1.1 / 4
        camarilla_l3[i] = pclose - range_val * 1.1 / 4
        camarilla_h4[i] = pclose + range_val * 1.1 / 2
        camarilla_l4[i] = pclose - range_val * 1.1 / 2
    
    # Calculate 12h ADX with min_periods
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # Directional Movement
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    period = 14
    atr = np.full(len(close_12h), np.nan)
    plus_dm_smooth = np.full(len(close_12h), np.nan)
    minus_dm_smooth = np.full(len(close_12h), np.nan)
    
    if len(close_12h) >= period + 1:
        # Initial values
        atr[period] = np.nanmean(tr[1:period+1])
        plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
        
        # Wilder's smoothing
        for i in range(period + 1, len(close_12h)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period - 1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period - 1) + minus_dm[i]) / period
    
    # Directional Indicators
    plus_di = np.full(len(close_12h), np.nan)
    minus_di = np.full(len(close_12h), np.nan)
    dx = np.full(len(close_12h), np.nan)
    
    for i in range(period, len(close_12h)):
        if atr[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # ADX (smoothed DX)
    adx = np.full(len(close_12h), np.nan)
    adx_period = 14
    if len(close_12h) >= 2 * period:
        # Initial ADX
        adx[2*period] = np.nanmean(dx[period+1:2*period+1])
        # Smoothed ADX
        for i in range(2*period + 1, len(close_12h)):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Get 6h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align 12h indicators to 6h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions at Camarilla H4/L4 levels
        long_breakout = close[i] > camarilla_h4_aligned[i]
        short_breakout = close[i] < camarilla_l4_aligned[i]
        
        # Trend filter from 12h ADX
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20  # Regime change to range
        
        # Entry logic: Breakout + strong trend + volume confirmation
        long_entry = long_breakout and strong_trend and volume_spike[i]
        short_entry = short_breakout and strong_trend and volume_spike[i]
        
        # Exit logic: 
        # 1. Price reverses to H3/L3 levels (profit taking/reversal)
        # 2. Trend weakens (ADX < 20) - range regime
        long_exit = (close[i] < camarilla_h3_aligned[i]) or weak_trend
        short_exit = (close[i] > camarilla_l3_aligned[i]) or weak_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_camarilla_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0