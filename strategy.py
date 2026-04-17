#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d ADX Trend Filter + Volume Spike.
Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending up) AND volume > 1.5x average.
Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending down) AND volume > 1.5x average.
Exit when Williams %R returns to -50 (mean reversion) OR ADX < 20 (trend weakens).
Uses 1d for ADX calculation, 6h for Williams %R and volume.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (Average Directional Index)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        plus_dm = np.zeros_like(close)
        minus_dm = np.zeros_like(close)
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, +DM, -DM (Wilder's smoothing)
        atr = np.zeros_like(close)
        atr_plus = np.zeros_like(close)
        atr_minus = np.zeros_like(close)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        atr_plus[period] = np.mean(plus_dm[1:period+1])
        atr_minus[period] = np.mean(minus_dm[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            atr_plus[i] = (atr_plus[i-1] * (period-1) + plus_dm[i]) / period
            atr_minus[i] = (atr_minus[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.zeros_like(close)
        minus_di = np.zeros_like(close)
        dx = np.zeros_like(close)
        
        for i in range(period, len(close)):
            if atr[i] > 0:
                plus_di[i] = 100 * atr_plus[i] / atr[i]
                minus_di[i] = 100 * atr_minus[i] / atr[i]
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
                else:
                    dx[i] = 0
            else:
                plus_di[i] = 0
                minus_di[i] = 0
                dx[i] = 0
        
        # ADX (smoothed DX)
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 6h Williams %R
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        
        wr = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if highest_high[i] - lowest_low[i] != 0:
                wr[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
            else:
                wr[i] = -50
        return wr
    
    wr_6h = calculate_williams_r(high, low, close, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume spike (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_6h[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr_val = wr_6h[i]
        adx_val = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: oversold + strong uptrend + volume spike
            if wr_val < -80 and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: overbought + strong downtrend + volume spike
            elif wr_val > -20 and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: mean reversion or trend weakening
            if wr_val > -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: mean reversion or trend weakening
            if wr_val < -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ADXTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0