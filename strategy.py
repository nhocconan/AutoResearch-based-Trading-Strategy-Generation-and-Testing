#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX Trend Filter + Volume Spike Confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and rising (momentum) AND 1d ADX > 25 (trending market) AND volume spike
# Short when Bear Power < 0 and falling (momentum) AND 1d ADX > 25 AND volume spike
# Uses 1d ADX to avoid ranging markets where Elder Ray gives false signals.
# Designed for 12-37 trades/year on 6h to minimize fee drag while capturing strong trends.
# Works in bull markets via long signals in uptrend and bear markets via short signals in downtrend.

name = "6h_ElderRay_1dADX_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX for trend filter (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed TR, DM+ and DM- (using Wilder's smoothing = EMA with alpha=1/period)
        tr_period = len(tr)
        atr = np.zeros(tr_period)
        dm_plus_smooth = np.zeros(tr_period)
        dm_minus_smooth = np.zeros(tr_period)
        
        # Initial values (simple average)
        atr[period-1] = np.mean(tr[:period])
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        # Wilder's smoothing
        for i in range(period, tr_period):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # ADX
        dx = np.abs(di_plus - di_minus) / (np.abs(di_plus + di_minus)) * 100
        adx = np.zeros_like(dx)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])  # First ADX value
        
        for i in range(2*period-1, tr_period):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate EMA13 for Elder Ray on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Elder Ray momentum (change from previous bar)
        if i > 0:
            bull_power_momentum = bull_power[i] - bull_power[i-1]
            bear_power_momentum = bear_power[i] - bear_power[i-1]
        else:
            bull_power_momentum = 0
            bear_power_momentum = 0
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND rising momentum AND 1d ADX > 25 AND volume spike
            if (bull_power[i] > 0 and 
                bull_power_momentum > 0 and 
                adx_1d_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND falling momentum AND 1d ADX > 25 AND volume spike
            elif (bear_power[i] < 0 and 
                  bear_power_momentum < 0 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR momentum turns negative OR ADX < 20 (ranging)
            if (bull_power[i] <= 0 or 
                bull_power_momentum <= 0 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR momentum turns positive OR ADX < 20 (ranging)
            if (bear_power[i] >= 0 or 
                bear_power_momentum >= 0 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals