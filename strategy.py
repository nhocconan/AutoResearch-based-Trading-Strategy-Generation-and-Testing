#!/usr/bin/env python3
"""
6h Modified Donchian + Volume Filter + ADX Trend Filter
Hypothesis: Uses 6h Donchian channel breakouts confirmed by volume spike and 
ADX trend strength (>25) to capture strong momentum moves. Includes ATR-based 
stop loss and time-based exit to prevent overtrading. Works in both bull and 
bear markets by capturing breakouts in direction of trend.
Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_moddonchian_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 6-period ADX for trend strength
    def calculate_adx(high, low, close, period=6):
        n = len(high)
        if n < period:
            return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        
        # Smooth TR, DM+
        tr_period = np.full(n, np.nan)
        dm_plus_period = np.full(n, np.nan)
        dm_minus_period = np.full(n, np.nan)
        
        if n >= period + 1:
            tr_period[period] = np.sum(tr[:period])
            dm_plus_period[period] = np.sum(dm_plus[:period])
            dm_minus_period[period] = np.sum(dm_minus[:period])
            
            for i in range(period + 1, n):
                tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i-1]
                dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / period) + dm_plus[i-1]
                dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / period) + dm_minus[i-1]
        
        # Directional Indicators
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if not np.isnan(tr_period[i]) and tr_period[i] != 0:
                plus_di[i] = 100 * dm_plus_period[i] / tr_period[i]
                minus_di[i] = 100 * dm_minus_period[i] / tr_period[i]
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX (smoothed DX)
        adx = np.full(n, np.nan)
        if n >= 2 * period:
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            for i in range(2*period, n):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
                else:
                    adx[i] = adx[i-1]
        
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=6)
    
    # 6-period Donchian channel (20 periods lookback for breakout)
    def calculate_donchian(high, low, period=20):
        n = len(high)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        
        if n >= period:
            for i in range(period-1, n):
                upper[i] = np.max(high[i-period+1:i+1])
                lower[i] = np.min(low[i-period+1:i+1])
        
        return upper, lower
    
    donch_high, donch_low = calculate_donchian(high, low, period=20)
    
    # Volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # Need enough data for all indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (current volume > 1.5x average)
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian low OR ADX weakens OR stoploss hit
            if (close[i] < donch_low[i] or
                adx[i] < 20 or  # Trend weakening
                close[i] < entry_price - 2.5 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian high OR ADX weakens OR stoploss hit
            if (close[i] > donch_high[i] or
                adx[i] < 20 or  # Trend weakening
                close[i] > entry_price + 2.5 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries - minimum 12 bars flat between trades
            if bars_since_entry >= 12:
                # Trend filter: ADX > 25 and DI crossover
                strong_trend = adx[i] > 25
                bullish_di = plus_di[i] > minus_di[i]
                bearish_di = minus_di[i] > plus_di[i]
                
                # Breakout entries with volume confirmation
                bull_breakout = close[i] > donch_high[i]
                bear_breakout = close[i] < donch_low[i]
                
                # Long: bullish breakout with strong uptrend and volume
                if bull_breakout and strong_trend and bullish_di and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout with strong downtrend and volume
                elif bear_breakout and strong_trend and bearish_di and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals