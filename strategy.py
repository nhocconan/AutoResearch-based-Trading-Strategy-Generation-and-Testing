#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla H3/L3 breakout with 1d ADX trend filter + volume confirmation
    # Long: price > Camarilla H3 + price > 1d ADX(14) > 25 + volume > 1.5x 20-period average
    # Short: price < Camarilla L3 + price < 1d ADX(14) > 25 + volume > 1.5x 20-period average
    # Exit: price crosses Camarilla H4/L4 OR ADX drops below 20
    # Using 4h timeframe for balance of signal quality and trade frequency, 1d Camarilla pivots for strong structure,
    # and ADX trend filter to avoid false breakouts in ranging markets.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4) with min_periods
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i < 1:  # Need at least 1 period for calculation
            continue
        # Use previous day's OHLC for today's Camarilla levels
        phigh = high_1d[i-1] if i-1 >= 0 else high_1d[0]
        plow = low_1d[i-1] if i-1 >= 0 else low_1d[0]
        pclose = close_1d[i-1] if i-1 >= 0 else close_1d[0]
        
        # Camarilla levels calculation
        rang = phigh - plow
        camarilla_h3[i] = pclose + rang * 1.1 / 4
        camarilla_l3[i] = pclose - rang * 1.1 / 4
        camarilla_h4[i] = pclose + rang * 1.1 / 2
        camarilla_l4[i] = pclose - rang * 1.1 / 2
    
    # Calculate 1d ADX(14) with min_periods
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
        tr = np.concatenate([[np.nan], tr])  # Align with original arrays
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = np.full(n, np.nan)
        dm_plus_smooth = np.full(n, np.nan)
        dm_minus_smooth = np.full(n, np.nan)
        
        # Initial values (simple average)
        if n >= period:
            atr[period-1] = np.nanmean(tr[1:period])  # Skip first NaN
            dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
            dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
            
            # Wilder's smoothing
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.full(n, np.nan)
        di_minus = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if atr[i] != 0:
                di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
                di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
                if (di_plus[i] + di_minus[i]) != 0:
                    dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
        
        # ADX (smoothed DX)
        adx = np.full(n, np.nan)
        if n >= 2*period:
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            for i in range(2*period, n):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align 1d indicators to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions at Camarilla H3/L3
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Trend filter from 1d ADX
        strong_trend = adx_1d_aligned[i] > 25
        
        # Entry logic: Breakout + strong trend + volume confirmation
        long_entry = long_breakout and strong_trend and volume_spike[i]
        short_entry = short_breakout and strong_trend and volume_spike[i]
        
        # Exit logic: Camarilla H4/L4 OR trend weakening (ADX < 20)
        long_exit = close[i] > camarilla_h4_aligned[i] or close[i] < camarilla_l4_aligned[i] or adx_1d_aligned[i] < 20
        short_exit = close[i] > camarilla_h4_aligned[i] or close[i] < camarilla_l4_aligned[i] or adx_1d_aligned[i] < 20
        
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

name = "4h_1d_camarilla_h3l3_breakout_adx_volume_v1"
timeframe = "4h"
leverage = 1.0