#!/usr/bin/env python3
"""
1-day Donchian(20) breakout with 1-week ADX filter and volume confirmation
Hypothesis: Breakouts of weekly Donchian channels in the direction of the weekly ADX trend (>25),
confirmed by daily volume > 1.5x 20-day average, capture major trends with minimal whipsaws.
Designed for ~15-25 trades/year to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_adx_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1-week ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.nanmean(tr[1:period])
        plus_dm_sum = np.nansum(plus_dm[1:period])
        minus_dm_sum = np.nansum(minus_dm[1:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
            
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_14_1w_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ADX falls below 20 (weak trend) OR price breaks below weekly Donchian(10) low
            donchian_low = np.min(low[max(0, i-10):i+1])
            if (adx_14_1w_aligned[i] < 20 or 
                close[i] <= donchian_low):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX falls below 20 (weak trend) OR price breaks above weekly Donchian(10) high
            donchian_high = np.max(high[max(0, i-10):i+1])
            if (adx_14_1w_aligned[i] < 20 or 
                close[i] >= donchian_high):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Weekly Donchian(20) channels
            donchian_high = np.max(high_1w[max(0, (i//7)-20):(i//7)]) if i >= 7*20 else np.max(high[:i])
            donchian_low = np.min(low_1w[max(0, (i//7)-20):(i//7)]) if i >= 7*20 else np.min(low[:i])
            
            # Align weekly levels to daily index
            if i >= 7*20:
                # Map daily index to weekly index
                weekly_idx = i // 7
                if weekly_idx < len(high_1w):
                    donchian_high = np.max(high_1w[max(0, weekly_idx-20):weekly_idx])
                    donchian_low = np.min(low_1w[max(0, weekly_idx-20):weekly_idx])
                else:
                    donchian_high = np.max(high_1w[:weekly_idx]) if weekly_idx > 0 else high_1w[0]
                    donchian_low = np.min(low_1w[:weekly_idx]) if weekly_idx > 0 else low_1w[0]
            else:
                donchian_high = np.max(high[:i]) if i > 0 else high[0]
                donchian_low = np.min(low[:i]) if i > 0 else low[0]
            
            # Long: price breaks above weekly Donchian(20) high + ADX > 25 + volume spike
            if (close[i] > donchian_high and
                adx_14_1w_aligned[i] > 25 and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly Donchian(20) low + ADX > 25 + volume spike
            elif (close[i] < donchian_low and
                  adx_14_1w_aligned[i] > 25 and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals