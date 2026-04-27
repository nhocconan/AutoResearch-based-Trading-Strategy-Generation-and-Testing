#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d ADX trend filter and volume confirmation.
# Donchian channel identifies breakouts, ADX confirms trend strength, volume confirms institutional participation.
# Long: price breaks above upper band + ADX > 25 + volume > 1.5x avg
# Short: price breaks below lower band + ADX > 25 + volume > 1.5x avg
# Exit: opposite breakout or ADX < 20 (trend weakening)
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX calculation (14-period)
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
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(high)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Get 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout logic with ADX trend filter and volume confirmation
        if adx_1d_aligned[i] > 25 and volume_filter[i]:  # Strong trend
            # Long breakout: price above upper Donchian band
            if close[i] > high_max[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price below lower Donchian band
            elif close[i] < low_min[i]:
                signals[i] = -0.25
                position = -1
            # Exit conditions: opposite breakout or trend weakening
            elif position == 1 and close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            elif position == 1 and adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            elif position == -1 and adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            # Hold position
            elif position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # Weak trend or low volume: stay flat or hold
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1dADX14_VolumeFilter"
timeframe = "12h"
leverage = 1.0