#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R + ADX Trend + Volume Spike. Long when Williams %R < -80 (oversold) with ADX > 25 and volume > 1.5x average in uptrend (close > EMA50). Short when Williams %R > -20 (overbought) with ADX > 25 and volume > 1.5x average in downtrend (close < EMA50). Exit when Williams %R returns to -50 or trend weakens (ADX < 20). Uses 1d for ADX calculation, 4h for price/volume/Williams %R. Target: 75-200 total trades over 4 years (19-50/year). Uses tighter volume threshold and trend filter to reduce trade frequency and improve edge in both bull and bear markets.
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
    
    # Calculate 14-period ADX
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(close)
        dm_minus = np.zeros_like(close)
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing)
        tr_period = np.zeros_like(close)
        dm_plus_period = np.zeros_like(close)
        dm_minus_period = np.zeros_like(close)
        
        # Initial values
        tr_period[period] = np.mean(tr[1:period+1])
        dm_plus_period[period] = np.mean(dm_plus[1:period+1])
        dm_minus_period[period] = np.mean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(close)):
            tr_period[i] = (tr_period[i-1] * (period-1) + tr[i]) / period
            dm_plus_period[i] = (dm_plus_period[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_period[i] = (dm_minus_period[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.zeros_like(close)
        di_minus = np.zeros_like(close)
        for i in range(period, len(close)):
            if tr_period[i] > 0:
                di_plus[i] = 100 * dm_plus_period[i] / tr_period[i]
                di_minus[i] = 100 * dm_minus_period[i] / tr_period[i]
            else:
                di_plus[i] = 0
                di_minus[i] = 0
        
        # DX and ADX
        dx = np.zeros_like(close)
        for i in range(period, len(close)):
            if di_plus[i] + di_minus[i] > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
            else:
                dx[i] = 0
        
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        
        williams_r = np.full_like(close, -50.0)  # default neutral
        for i in range(period-1, len(close)):
            if highest_high[i] - lowest_low[i] != 0:
                williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
            else:
                williams_r[i] = -50.0
        return williams_r
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Calculate EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume spike (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema50[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        williams_r_val = williams_r[i]
        adx_val = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        ema50_val = ema50[i]
        
        # Trend filters: ADX > 25 = strong trend, ADX < 20 = weak trend/no trend
        strong_trend = adx_val > 25
        weak_trend = adx_val < 20
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with strong uptrend (close > EMA50) and volume spike
            if williams_r_val < -80 and price > ema50_val and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with strong downtrend (close < EMA50) and volume spike
            elif williams_r_val > -20 and price < ema50_val and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR trend weakens
            if williams_r_val >= -50 or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR trend weakens
            if williams_r_val <= -50 or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_ADXTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0