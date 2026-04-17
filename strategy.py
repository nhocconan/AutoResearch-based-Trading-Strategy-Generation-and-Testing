#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d ADX Trend Filter + Volume Confirmation.
Long when Williams %R < -80 (oversold) in a strong uptrend (ADX > 25) with volume spike.
Short when Williams %R > -20 (overbought) in a strong downtrend (ADX > 25) with volume spike.
Exit when Williams %R returns to -50 (mean reversion) or trend weakens (ADX < 20).
Uses 1d for Williams %R and ADX calculation, 6h for price/volume.
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
    
    # Get 1d data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (%R = (Highest High - Close) / (Highest High - Lowest Low) * -100)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        williams_r = np.full_like(close, -50.0)  # neutral default
        
        for i in range(len(close)):
            if i >= period - 1:
                start_idx = i - period + 1
                highest_high[i] = np.max(high[start_idx:i+1])
                lowest_low[i] = np.min(low[start_idx:i+1])
                if highest_high[i] != lowest_low[i]:
                    williams_r[i] = ((highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])) * -100
                else:
                    williams_r[i] = -50.0
        return williams_r
    
    # Calculate 1d ADX (Average Directional Index)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
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
        plus_dm_smooth = np.zeros_like(close)
        minus_dm_smooth = np.zeros_like(close)
        
        # Initial values (simple average)
        if len(close) >= period:
            atr[period-1] = np.mean(tr[1:period])
            plus_dm_smooth[period-1] = np.mean(plus_dm[1:period])
            minus_dm_smooth[period-1] = np.mean(minus_dm[1:period])
        
        # Wilder's smoothing
        for i in range(period, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.zeros_like(close)
        minus_di = np.zeros_like(close)
        dx = np.zeros_like(close)
        
        for i in range(period, len(close)):
            if atr[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
                else:
                    dx[i] = 0
            else:
                plus_di[i] = 0
                minus_di[i] = 0
                dx[i] = 0
        
        # ADX (smoothed DX)
        adx = np.full_like(close, 0.0)
        if len(close) >= 2 * period - 1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(close)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    williams_r_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume spike (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        williams_r = williams_r_1d_aligned[i]
        adx = adx_1d_aligned[i]
        
        # Trend regime: ADX > 25 = strong trend (good for momentum)
        is_strong_trend = adx > 25
        # Weak trend: ADX < 20 (avoid false signals)
        is_weak_trend = adx < 20
        
        if position == 0:
            # Long: Williams %R oversold (< -80) in strong uptrend with volume spike
            if williams_r < -80 and is_strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) in strong downtrend with volume spike
            elif williams_r > -20 and is_strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR trend weakens
            if williams_r >= -50 or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR trend weakens
            if williams_r <= -50 or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ADXTrend_Volume"
timeframe = "6h"
leverage = 1.0