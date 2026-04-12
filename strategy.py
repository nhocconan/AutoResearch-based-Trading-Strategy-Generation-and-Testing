#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d ADX regime filter
    # Uses 1d ADX > 25 to identify trending markets (bull/bear)
    # In trending markets: Elder Ray Bull Power (high - EMA13) > 0 for long, Bear Power (EMA13 - low) > 0 for short
    # In ranging markets (ADX <= 25): fade extreme Elder Ray readings (mean reversion)
    # Discrete sizing 0.25 to minimize fee churn. Target: 12-30 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for regime filter and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA13 for Elder Ray power calculation
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX for regime filter (trending vs ranging)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = WilderSmoothing(tr, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmoothing(dx, 14)
    
    # Align 1d indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Elder Ray Power on 6s timeframe using 1d EMA13
    # Bull Power = high - EMA13
    # Bear Power = EMA13 - low
    bull_power = high - ema13_1d_aligned
    bear_power = ema13_1d_aligned - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime: ADX > 25 = trending, ADX <= 25 = ranging
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] <= 25
        
        # Entry logic based on regime
        long_entry = False
        short_entry = False
        
        if trending:
            # Trending market: follow Elder Ray power
            long_entry = bull_power[i] > 0  # Bullish power positive
            short_entry = bear_power[i] > 0  # Bearish power positive
        else:
            # Ranging market: mean reversion at extreme Elder Ray readings
            # Use 20-period z-score of Elder Ray for mean reversion signals
            if i >= 70:  # Need enough data for 20-period lookback
                bull_ma = np.mean(bull_power[i-20:i])
                bull_std = np.std(bull_power[i-20:i]) if np.std(bull_power[i-20:i]) > 0 else 1
                bear_ma = np.mean(bear_power[i-20:i])
                bear_std = np.std(bear_power[i-20:i]) if np.std(bear_power[i-20:i]) > 0 else 1
                
                bull_z = (bull_power[i] - bull_ma) / bull_std
                bear_z = (bear_power[i] - bear_ma) / bear_std
                
                # Fade extreme readings: short when bull power extremely high, long when bear power extremely high
                long_entry = bear_z > 2.0  # Extreme bear power - expect reversion up
                short_entry = bull_z > 2.0  # Extreme bull power - expect reversion down
        
        # Exit logic: opposite signal or regime change
        long_exit = False
        short_exit = False
        
        if trending:
            # In trending market, exit when power reverses
            long_exit = bear_power[i] > 0  # Bearish power appears
            short_exit = bull_power[i] > 0  # Bullish power appears
        else:
            # In ranging market, exit when power normalizes
            if i >= 70:
                long_exit = bear_z < 0.5  # Bear power normalizes
                short_exit = bull_z < 0.5  # Bull power normalizes
            else:
                long_exit = True
                short_exit = True
        
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

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0