#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1w ADX(14) trend filter + volume confirmation
    # Long: price > Donchian(20) high + weekly ADX > 25 + volume > 2.0x 20-period average
    # Short: price < Donchian(20) low + weekly ADX > 25 + volume > 2.0x 20-period average
    # Exit: opposite Donchian breakout OR weekly ADX < 20 (trend weakening)
    # Using 6h timeframe for balance of signal quality and trade frequency, weekly ADX for strong trend filter,
    # and volume spike confirmation to avoid false breakouts in choppy markets.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX(14) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 28:  # Need 2*period for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ADX(14) with min_periods
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period + 1:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(np.abs(high[1:] - close[:-1]), 
                                   np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[np.nan], tr])  # Align with indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values
        atr = np.full(n, np.nan)
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        
        if n >= period:
            # Initial smoothed values (simple average)
            atr[period-1] = np.nanmean(tr[1:period+1])
            plus_dm_avg = np.nanmean(plus_dm[1:period+1])
            minus_dm_avg = np.nanmean(minus_dm[1:period+1])
            
            if atr[period-1] != 0:
                plus_di[period-1] = (plus_dm_avg / atr[period-1]) * 100
                minus_di[period-1] = (minus_dm_avg / atr[period-1]) * 100
            
            # Wilder's smoothing
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
                plus_dm_avg = (plus_dm_avg * (period - 1) + plus_dm[i]) / period
                minus_dm_avg = (minus_dm_avg * (period - 1) + minus_dm[i]) / period
                if atr[i] != 0:
                    plus_di[i] = (plus_dm_avg / atr[i]) * 100
                    minus_di[i] = (minus_dm_avg / atr[i]) * 100
        
        # Calculate DX and ADX
        dx = np.full(n, np.nan)
        adx = np.full(n, np.nan)
        
        if n >= 2*period-1:
            for i in range(period, n):
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
            
            if n >= 2*period:
                adx[2*period-1] = np.nanmean(dx[period:2*period])
                for i in range(2*period, n):
                    adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Get 6h Donchian(20) for breakout with min_periods
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get 6h volume for confirmation (>2.0x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align weekly ADX to 6h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Trend filter from weekly ADX
        strong_trend = adx_1w_aligned[i] > 25
        weakening_trend = adx_1w_aligned[i] < 20
        
        # Entry logic: Breakout + strong trend + volume confirmation
        long_entry = long_breakout and strong_trend and volume_spike[i]
        short_entry = short_breakout and strong_trend and volume_spike[i]
        
        # Exit logic: opposite breakout or trend weakening
        long_exit = short_breakout or weakening_trend
        short_exit = long_breakout or weakening_trend
        
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

name = "6h_1w_donchian_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0