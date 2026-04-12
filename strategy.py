#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + ADX regime filter
    # Uses 1d Elder Ray (Bull/Bear Power) for trend strength and 1d ADX for regime
    # Long when Bull Power > 0 and ADX > 25 (strong uptrend)
    # Short when Bear Power < 0 and ADX > 25 (strong downtrend)
    # Avoids choppy markets (ADX < 20) where trend following fails
    # Discrete sizing 0.25 to minimize fee churn. Target: 15-35 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Elder Ray and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13 = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        ema13[i] = np.mean(close_1d[i-12:i+1])
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Calculate 1d ADX
    # True Range
    tr = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]), 
                    abs(low_1d[i] - close_1d[i-1]))
    
    # +DM and -DM
    plus_dm = np.full(len(close_1d), np.nan)
    minus_dm = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_period = 14
    tr_smooth = WilderSmoothing(tr, atr_period)
    plus_dm_smooth = WilderSmoothing(plus_dm, atr_period)
    minus_dm_smooth = WilderSmoothing(minus_dm, atr_period)
    
    # +DI and -DI
    plus_di = np.full(len(close_1d), np.nan)
    minus_di = np.full(len(close_1d), np.nan)
    dx = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if tr_smooth[i] > 0:
            plus_di[i] = (plus_dm_smooth[i] / tr_smooth[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / tr_smooth[i]) * 100
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # ADX: smoothed DX
    adx = WilderSmoothing(dx, atr_period)
    
    # Align to 6h timeframe (use previous day's values)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market (good for trend following)
        trending_market = adx_aligned[i] > 25
        
        # Entry conditions: Elder Ray signals in trending market
        long_entry = bull_power_aligned[i] > 0 and trending_market
        short_entry = bear_power_aligned[i] < 0 and trending_market
        
        # Exit conditions: opposing Elder Ray signal or ADX weakening
        long_exit = bear_power_aligned[i] < 0 or adx_aligned[i] < 20
        short_exit = bull_power_aligned[i] > 0 or adx_aligned[i] < 20
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0