#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w ADX trend filter and volume confirmation.
# Donchian channels provide clear breakout signals. ADX confirms trend strength on weekly.
# Volume ensures breakouts have institutional backing. Works in bull markets (breakouts continue)
# and bear markets (breakdowns from strength). Target: 20-50 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w data for ADX trend filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ADX calculation on weekly data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = (plus_dm_sum * (period-1) + plus_dm[i]) / period
            minus_dm_sum = (minus_dm_sum * (period-1) + minus_dm[i]) / period
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        for i in range(period*2, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[2*period:3*period+1]) if len(dx) > 3*period else 0
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === 12h data for Donchian breakout and volume ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = np.zeros_like(high_12h)
    donchian_low = np.zeros_like(low_12h)
    
    for i in range(len(high_12h)):
        if i >= 19:
            donchian_high[i] = np.max(high_12h[i-19:i+1])
            donchian_low[i] = np.min(low_12h[i-19:i+1])
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume average (20-period)
    vol_avg20_12h = np.zeros_like(volume_12h)
    for i in range(len(volume_12h)):
        if i >= 19:
            vol_avg20_12h[i] = np.mean(volume_12h[i-19:i+1])
        else:
            vol_avg20_12h[i] = np.nan
    
    vol_avg20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg20_12h)
    
    signals = np.zeros(n)
    position = 0
    warmup = 100  # Sufficient for all indicators
    
    for i in range(warmup, n):
        if (np.isnan(adx_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_avg20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        vol_filter = vol_12h_current > 1.5 * vol_avg20_12h_aligned[i]
        
        if position == 0:
            # Long: breakout above Donchian high + strong trend (ADX > 25) + volume
            if close[i] > donchian_high_aligned[i] and adx_1w_aligned[i] > 25 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + strong trend (ADX > 25) + volume
            elif close[i] < donchian_low_aligned[i] and adx_1w_aligned[i] > 25 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakdown below Donchian low or trend weakening (ADX < 20)
            if close[i] < donchian_low_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above Donchian high or trend weakening (ADX < 20)
            if close[i] > donchian_high_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_1wADX_VolumeFilter"
timeframe = "12h"
leverage = 1.0