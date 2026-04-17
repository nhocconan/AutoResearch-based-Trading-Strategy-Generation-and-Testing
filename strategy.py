#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with daily ADX trend filter and volume confirmation.
# Donchian channels capture breakouts from volatility contractions. ADX > 25 ensures
# we only trade in trending markets, avoiding whipsaws in ranges. Volume confirmation
# ensures breakouts have conviction. This combination works in both bull and bear
# markets by capturing strong directional moves while filtering false breakouts.
# Target: 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for ADX trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        # Smoothed values
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        if len(high) > period:
            # Initial averages
            atr[period] = np.mean(tr[1:period+1])
            plus_dm_sum = np.sum(plus_dm[1:period+1])
            minus_dm_sum = np.sum(minus_dm[1:period+1])
            plus_di[period] = 100 * plus_dm_sum / atr[period] if atr[period] > 0 else 0
            minus_di[period] = 100 * minus_dm_sum / atr[period] if atr[period] > 0 else 0
            
            # Smooth subsequent values
            for i in range(period+1, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_di[i] = 100 * ((plus_di[i-1] * (period-1)) + plus_dm[i]) / (atr[i] * period) if atr[i] > 0 else 0
                minus_di[i] = 100 * ((minus_di[i-1] * (period-1)) + minus_dm[i]) / (atr[i] * period) if atr[i] > 0 else 0
        
        # Calculate DX and ADX
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        if len(high) > 2*period:
            adx[2*period] = np.mean(dx[period:2*period+1])
            for i in range(2*period+1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Donchian channels (20-period) ===
    donch_period = 20
    donch_high = np.zeros_like(high)
    donch_low = np.zeros_like(low)
    
    for i in range(donch_period-1, len(high)):
        donch_high[i] = np.max(high[i-donch_period+1:i+1])
        donch_low[i] = np.min(low[i-donch_period+1:i+1])
    
    # === 6h volume confirmation (20-period average) ===
    vol_avg20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_avg20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    warmup = max(100, donch_period*2, 30)  # Sufficient warmup
    
    for i in range(warmup, n):
        if np.isnan(adx_1d_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_avg20[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average
        vol_filter = volume[i] > 1.5 * vol_avg20[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: break above upper Donchian band + trend + volume
            if close[i] > donch_high[i] and trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian band + trend + volume
            elif close[i] < donch_low[i] and trend_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian band or ADX weakens
            if close[i] < donch_low[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian band or ADX weakens
            if close[i] > donch_high[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_ADX_VolumeFilter"
timeframe = "6h"
leverage = 1.0