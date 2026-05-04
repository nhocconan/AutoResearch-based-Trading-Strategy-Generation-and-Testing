#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter + volume confirmation
# Long when Bull Power > 0, Bear Power < 0, 1d ADX > 25 (trending), and volume > 1.5x 20-period volume EMA
# Short when Bear Power > 0, Bull Power < 0, 1d ADX > 25 (trending), and volume > 1.5x 20-period volume EMA
# Exit when Elder Ray signals reverse or ADX < 20 (range regime)
# Uses 1d HTF ADX for regime filter to avoid whipsaw in ranging markets, targeting 12-37 trades/year on 6h.
# Volume confirmation (1.5x) reduces false breakouts. Works in bull markets via longs in bullish Elder Ray + ADX>25
# and bear markets via shorts in bearish Elder Ray + ADX>25.

name = "6h_ElderRay_ADXRegime_VolumeConfirm"
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
    
    # Get 1d data for HTF ADX and Elder Ray components - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        tr_period = len(tr)
        atr = np.full(tr_period, np.nan)
        plus_dm_smooth = np.full(tr_period, np.nan)
        minus_dm_smooth = np.full(tr_period, np.nan)
        
        # Initial values (simple average)
        if tr_period >= period:
            atr[period-1] = np.nanmean(tr[1:period])  # skip first NaN
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period])
            
            # Wilder's smoothing
            for i in range(period, tr_period):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full(tr_period, np.nan)
        minus_di = np.full(tr_period, np.nan)
        dx = np.full(tr_period, np.nan)
        
        for i in range(period, tr_period):
            if atr[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # ADX: smoothed DX
        adx = np.full(tr_period, np.nan)
        if tr_period >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])  # initial ADX
            for i in range(2*period-1, tr_period):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray components (13-period EMA for reference)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, ADX > 25, volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                adx_1d_aligned[i] > 25 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0, Bull Power < 0, ADX > 25, volume spike
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  adx_1d_aligned[i] > 25 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Elder Ray reverses OR ADX < 20 (range regime)
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Elder Ray reverses OR ADX < 20 (range regime)
            if (bear_power[i] <= 0 or bull_power[i] >= 0 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals