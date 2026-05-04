#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Volume Spike with 1d ADX Regime Filter
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# Long when Bull Power > 0 AND rising AND volume spike in bullish 1d ADX > 25 regime
# Short when Bear Power > 0 AND rising AND volume spike in bearish 1d ADX > 25 regime
# Uses 1d ADX to only trade in strong trends, avoiding whipsaws in ranging markets
# Volume spike confirms institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Target: 12-30 trades/year (50-120 total over 4 years) to overcome fee drag.

name = "6h_ElderRay_VolumeSpike_1dADX25_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14) for regime filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
        tr_period = len(tr)
        atr = np.full(tr_period, np.nan)
        plus_dm_smooth = np.full(tr_period, np.nan)
        minus_dm_smooth = np.full(tr_period, np.nan)
        
        # First values: simple average
        if tr_period >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period])
            
            # Wilder's smoothing
            for i in range(period, tr_period):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.full(tr_period, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.full(tr_period, np.nan)
        
        # First ADX: simple average of DX
        if tr_period >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            # Wilder's smoothing for ADX
            for i in range(2*period-1, tr_period):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Slope of Elder Ray (rising trend confirmation)
    bull_power_slope = bull_power - np.roll(bull_power, 1)
    bear_power_slope = bear_power - np.roll(bear_power, 1)
    bull_power_slope[0] = np.nan
    bear_power_slope[0] = np.nan
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_slope[i]) or np.isnan(bear_power_slope[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when 1d ADX > 25 (strong trend)
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND rising AND volume spike in strong bullish trend
            if (bull_power[i] > 0 and bull_power_slope[i] > 0 and 
                volume[i] > (2.0 * vol_ema_20[i]) and strong_trend):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0 AND rising AND volume spike in strong bearish trend
            elif (bear_power[i] > 0 and bear_power_slope[i] > 0 and 
                  volume[i] > (2.0 * vol_ema_20[i]) and strong_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR falling OR weak trend
            if (bull_power[i] <= 0 or bull_power_slope[i] <= 0 or not strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power <= 0 OR falling OR weak trend
            if (bear_power[i] <= 0 or bear_power_slope[i] <= 0 or not strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals