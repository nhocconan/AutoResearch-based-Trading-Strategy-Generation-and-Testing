#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with Volume and ADX Filter
# Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirms institutional participation.
# ADX > 25 filters for trending markets, avoiding whipsaws in ranging conditions.
# Works in both bull and bear markets: breaks above upper band in bull, breaks below lower band in bear.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_donchian_breakout_volume_adx_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # ADX (14-period) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            else:
                plus_dm[i] = 0
                
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
            else:
                minus_dm[i] = 0
                
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Smooth TR and DM
        atr = np.zeros_like(high)
        atr[period-1] = np.mean(tr[1:period])
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * np.zeros_like(high)
        minus_di = 100 * np.zeros_like(high)
        dx = np.zeros_like(high)
        
        plus_dm_smoothed = np.zeros_like(high)
        minus_dm_smoothed = np.zeros_like(high)
        plus_dm_smoothed[period-1] = np.sum(plus_dm[1:period])
        minus_dm_smoothed[period-1] = np.sum(minus_dm[1:period])
        
        for i in range(period, len(high)):
            plus_dm_smoothed[i] = plus_dm_smoothed[i-1] - (plus_dm_smoothed[i-1] / period) + plus_dm[i]
            minus_dm_smoothed[i] = minus_dm_smoothed[i-1] - (minus_dm_smoothed[i-1] / period) + minus_dm[i]
            
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_smoothed[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smoothed[i] / atr[i]
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low or trend weakens
            if close[i] < donchian_low[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high or trend weakens
            if close[i] > donchian_high[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and trend
            if close[i] > donchian_high[i] and vol_filter[i] and adx[i] > 25:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and trend
            elif close[i] < donchian_low[i] and vol_filter[i] and adx[i] > 25:
                position = -1
                signals[i] = -0.25
    
    return signals