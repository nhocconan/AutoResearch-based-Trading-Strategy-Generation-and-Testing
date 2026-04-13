#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d ADX regime
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    # Regime: ADX(14) > 25 = trending, < 20 = ranging (hysteresis)
    # In trending: go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
    # In ranging: fade extremes - long when Bull Power < -std, short when Bear Power < -std
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years.
    # Works in bull (trend continuation) and bear (mean reversion in ranges).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for regime detection
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray on 6h timeframe
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Higher = stronger bulls
    bear_power = ema_13 - low   # Higher = stronger bears
    
    # Calculate standard deviation of power for z-score normalization
    bull_power_std = pd.Series(bull_power).rolling(window=50, min_periods=50).std().values
    bear_power_std = pd.Series(bear_power).rolling(window=50, min_periods=50).std().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_std[i]) or np.isnan(bear_power_std[i]) or bull_power_std[i] == 0 or bear_power_std[i] == 0):
            signals[i] = 0.0
            continue
        
        # Regime detection with hysteresis
        adx = adx_1d_aligned[i]
        # Trending: ADX > 25, Ranging: ADX < 20
        if adx > 25:
            regime = 'trending'
        elif adx < 20:
            regime = 'ranging'
        else:
            regime = 'transition'  # hold previous regime or stay flat
        
        # Normalize power scores
        bull_power_z = bull_power[i] / bull_power_std[i]
        bear_power_z = bear_power[i] / bear_power_std[i]
        
        # Entry/exit logic based on regime
        if regime == 'trending':
            # In trending markets, follow the stronger power
            long_entry = bull_power_z > 0.5 and bull_power_z > bear_power_z and position != 1
            short_entry = bear_power_z > 0.5 and bear_power_z > bull_power_z and position != -1
            exit_long = position == 1 and (bull_power_z < 0 or bear_power_z > bull_power_z)
            exit_short = position == -1 and (bear_power_z < 0 or bull_power_z > bear_power_z)
        elif regime == 'ranging':
            # In ranging markets, fade extremes (mean reversion)
            long_entry = bull_power_z < -1.0 and position != 1  # oversold bulls
            short_entry = bear_power_z < -1.0 and position != -1  # oversold bears
            exit_long = position == 1 and bull_power_z > -0.5
            exit_short = position == -1 and bear_power_z > -0.5
        else:
            # Transition regime - stay flat
            long_entry = False
            short_entry = False
            exit_long = position == 1
            exit_short = position == -1
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0