#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h ADX trend filter + volume confirmation
# Uses Donchian channel (20) for breakout signals, 12h ADX > 25 for trend strength,
# and volume > 1.5x 20-period average for confirmation. Designed for fewer trades
# with strong edge in trending markets. Works in both bull and bear by following
# the 12h trend direction. Target: 20-50 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 14-period ADX on 12h timeframe
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.zeros_like(tr)
        atr[period] = np.nansum(tr[1:period+1]) / period
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] > 0:
                plus_di[i] = 100 * (np.nansum(plus_dm[i-period+1:i+1]) / period) / atr[i]
                minus_di[i] = 100 * (np.nansum(minus_dm[i-period+1:i+1]) / period) / atr[i]
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nansum(dx[period:2*period]) / period
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate Donchian channel (20) on 4h
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend strength
        is_strong_trend = adx_12h_aligned[i] > 25
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian upper band in strong trend with volume
            long_signal = False
            if is_strong_trend and has_volume and price > highest_high[i]:
                long_signal = True
            
            # Enter short: price breaks below Donchian lower band in strong trend with volume
            short_signal = False
            if is_strong_trend and has_volume and price < lowest_low[i]:
                short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian lower band OR ADX weakens
            exit_signal = False
            if price < lowest_low[i] or adx_12h_aligned[i] < 20:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper band OR ADX weakens
            exit_signal = False
            if price > highest_high[i] or adx_12h_aligned[i] < 20:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ADX12h_VolumeFilter"
timeframe = "4h"
leverage = 1.0