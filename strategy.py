#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + volume spike with 12h/1d regime filter
# - TRIX(15) for momentum: long when TRIX crosses above zero, short when crosses below zero
# - Volume spike: current 4h volume > 2.0x 20-period 4h average for conviction
# - Regime filter: use 12h ADX(14) > 25 to confirm trending market, avoid ranging
# - Exit on opposite TRIX cross or ADX drop below 20
# - Designed to capture momentum in trending markets while avoiding chop
# - Target: 25-40 trades/year to minimize fee drag

name = "4h_TRIX_VolumeSpike_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ADX(14) on 12h data
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        plus_dm = np.zeros_like(high_arr)
        minus_dm = np.zeros_like(high_arr)
        tr = np.zeros_like(high_arr)
        
        for i in range(1, len(high_arr)):
            plus_dm[i] = max(0, high_arr[i] - high_arr[i-1])
            minus_dm[i] = max(0, low_arr[i-1] - low_arr[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
                
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]),
                       abs(low_arr[i] - close_arr[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to alpha=1/period)
        atr = np.zeros_like(high_arr)
        plus_di = np.zeros_like(high_arr)
        minus_di = np.zeros_like(high_arr)
        dx = np.zeros_like(high_arr)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high_arr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # ADX is smoothed DX
        adx = np.zeros_like(high_arr)
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high_arr)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # TRIX(15) on 4h close
    def calculate_trix(close_arr, period=15):
        # Triple EMA
        ema1 = pd.Series(close_arr).ewm(span=period, adjust=False).values
        ema2 = pd.Series(ema1).ewm(span=period, adjust=False).values
        ema3 = pd.Series(ema2).ewm(span=period, adjust=False).values
        # TRIX = percent change of ema3
        trix = np.zeros_like(close_arr)
        trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
        return trix
    
    trix = calculate_trix(close, 15)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma.values * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if np.isnan(adx_12h_aligned[i]) or np.isnan(trix[i]) or np.isnan(vol_ma.iloc[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Look for long entry: TRIX crosses above zero + volume spike + ADX > 25 (trending)
            if i > 0 and trix[i-1] <= 0 and trix[i] > 0 and volume_spike[i] and adx_12h_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Look for short entry: TRIX crosses below zero + volume spike + ADX > 25 (trending)
            elif i > 0 and trix[i-1] >= 0 and trix[i] < 0 and volume_spike[i] and adx_12h_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on TRIX cross below zero or ADX drop below 20 (ranging)
            if trix[i] < 0 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on TRIX cross above zero or ADX drop below 20 (ranging)
            if trix[i] > 0 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals