#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX(14) trend filter and volume confirmation
# Donchian breakouts capture breakout momentum. ADX > 25 filters for trending markets.
# Volume > 1.5x 20-period average confirms breakout strength.
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for ADX trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on 1w data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.nansum(tr[1:period+1]) if np.any(~np.isnan(tr[1:period+1])) else 0
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = (np.nansum(plus_dm[i-period+1:i+1]) / atr[i]) * 100
                minus_di[i] = (np.nansum(minus_dm[i-period+1:i+1]) / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nansum(dx[period:2*period]) / period if np.any(~np.isnan(dx[period:2*period])) else 0
        
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian high + ADX > 25 + volume spike
            if (close[i] > donchian_high[i] and 
                adx_1w_aligned[i] > 25 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low + ADX > 25 + volume spike
            elif (close[i] < donchian_low[i] and 
                  adx_1w_aligned[i] > 25 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back through Donchian channel or ADX < 20
            if position == 1:
                # Exit long: Price below Donchian low or ADX < 20
                if close[i] < donchian_low[i] or adx_1w_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price above Donchian high or ADX < 20
                if close[i] > donchian_high[i] or adx_1w_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wADX25_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0