#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX25 regime filter and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0, 1d ADX > 25 (trending), and volume > 1.5x 20-bar average.
# Short when Bear Power < 0, Bull Power > 0, 1d ADX > 25 (trending), and volume > 1.5x average.
# Exit when Elder Power signals reverse or ADX drops below 20 (range regime).
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 6h timeframe.
# Elder Ray measures bull/bear strength via EMA13; ADX filters for trending markets only to avoid whipsaws.
# Volume spike confirms institutional participation. Works in bull/bear regimes by only trading strong trends.

name = "6h_ElderRay_1dADX25_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX25 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        # Smooth +DM and -DM
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
        for i in range(period+1, len(high)):
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Calculate +DI and -DI
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # Calculate ADX (smoothed DX)
        adx = np.zeros_like(high)
        adx[2*period] = np.nanmean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 6h timeframe (wait for 1d bar to close)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for volume avg
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0, Bear Power < 0, ADX > 25, volume > 1.5x avg
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx_1d_aligned[i] > 25 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0, Bull Power > 0, ADX > 25, volume > 1.5x avg
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  adx_1d_aligned[i] > 25 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power >= 0 (bulls weakening) OR ADX < 20 (ranging)
            if bear_power[i] >= 0 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power <= 0 (bears weakening) OR ADX < 20 (ranging)
            if bull_power[i] <= 0 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals