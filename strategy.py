#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# - Primary: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low)
# - Long: Bull Power > 0 and rising + ADX > 25 (trending market)
# - Short: Bear Power > 0 and rising + ADX > 25 (trending market)
# - Exit: Opposite Elder Ray signal or ADX < 20 (range regime)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# - Works in bull/bear: Elder Ray captures momentum strength, ADX filters chop, dual conditions reduce whipsaws

name = "6h_1d_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Elder Ray on 6h: EMA13 of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Smooth Elder Ray signals (3-period EMA) to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Calculate 1d ADX(14) for regime filter
    high_diff = high_1d - np.roll(high_1d, 1)
    low_diff = np.roll(low_1d, 1) - low_1d
    high_diff[0] = 0
    low_diff[0] = 0
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = np.where(atr_14 > 0, 100 * plus_dm_14 / atr_14, 0)
    minus_di = np.where(atr_14 > 0, 100 * minus_dm_14 / atr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray momentum: rising slope (current > previous)
        bull_rising = i > 0 and bull_power_smooth[i] > bull_power_smooth[i-1]
        bear_rising = i > 0 and bear_power_smooth[i] > bear_power_smooth[i-1]
        
        # Regime filter: ADX > 25 for trending, < 20 for ranging (hysteresis)
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 and rising + trending market
            if bull_power_smooth[i] > 0 and bull_rising and trending:
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power > 0 and rising + trending market
            elif bear_power_smooth[i] > 0 and bear_rising and trending:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Opposite Elder Ray signal (loss of momentum)
            # 2. ADX drops below 20 (regime shift to ranging)
            if position == 1:  # Long position
                exit_signal = (bull_power_smooth[i] <= 0) or (not bull_rising) or ranging
                if exit_signal:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_signal = (bear_power_smooth[i] <= 0) or (not bear_rising) or ranging
                if exit_signal:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals