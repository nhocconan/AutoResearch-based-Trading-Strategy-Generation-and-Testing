#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1-day ADX trend filter
# Uses 20-period Donchian channels for breakout detection, volume spike confirms breakout strength
# 1-day ADX > 25 filters for trending markets to avoid choppy conditions
# Trades only in direction of 1-day trend to reduce counter-trend losses
# Target: 20-35 trades/year per symbol (80-140 total over 4 years) to avoid fee drag
# Works in bull/bear markets by only trading with the trend on higher timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1-day data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            
            # If both are positive, the smaller one becomes zero
            if plus_dm[i] > 0 and minus_dm[i] > 0:
                if plus_dm[i] > minus_dm[i]:
                    minus_dm[i] = 0
                else:
                    plus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = (plus_dm_sum * (period-1) + plus_dm[i]) / period
            minus_dm_sum = (minus_dm_sum * (period-1) + minus_dm[i]) / period
            
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_sum / atr[i]
                minus_di[i] = 100 * minus_dm_sum / atr[i]
        
        # Calculate DX and ADX
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        for i in range(2*period, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # Wilder's smoothing for ADX
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 20-period Donchian channels on 4-hour data
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-(period-1):i+1])
            lower[i] = np.min(low[i-(period-1):i+1])
            
        return upper, lower
    
    donch_upper, donch_lower = donchian_channels(high, low, 20)
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4-hour timeframe
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(adx_14_1d_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper + volume spike + ADX > 25 (trending)
            if (close[i] > donch_upper[i] and vol_spike[i] and adx_14_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + volume spike + ADX > 25 (trending)
            elif (close[i] < donch_lower[i] and vol_spike[i] and adx_14_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < donch_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donch_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_Spike_ADX25_Trend"
timeframe = "4h"
leverage = 1.0