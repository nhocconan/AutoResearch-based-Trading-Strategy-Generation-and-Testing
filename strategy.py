#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Regime Filter
# - Bull Power = High - EMA13(1d), Bear Power = EMA13(1d) - Low
# - Long when Bull Power > 0 AND Bear Power rising (improving) AND 1d ADX > 25 (strong trend)
# - Short when Bear Power > 0 AND Bull Power falling (weakening) AND 1d ADX > 25 (strong trend)
# - Exit when power signals reverse or ADX < 20 (trend weakening)
# - Uses 1d for regime/trend strength and Elder Ray calculation, 6h for entry timing
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# - Elder Ray captures bull/bear strength via price relative to EMA, effective in both bull/bear markets
# - ADX filter ensures we only trade strong trends, avoiding whipsaws in ranging markets
# - Uses previous completed 1d bar for EMA/ADX to avoid look-ahead

name = "6h_1d_elder_ray_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for EMA13 and ADX
        return signals
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power = ema13_1d - low_1d   # Bear Power = EMA13 - Low
    
    # Calculate 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth DM and TR
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean()
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Align all 1d indicators to 6h timeframe with proper delay
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Track previous power values for momentum
    prev_bull_power = np.full(n, np.nan)
    prev_bear_power = np.full(n, np.nan)
    
    for i in range(30, n):  # Start from 30 to have sufficient lookback
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Store previous power values for momentum calculation
        if i > 0:
            prev_bull_power[i] = bull_power_aligned[i-1]
            prev_bear_power[i] = bear_power_aligned[i-1]
        else:
            prev_bull_power[i] = bull_power_aligned[i]
            prev_bear_power[i] = bear_power_aligned[i]
        
        # Power momentum: rising/falling
        bull_power_rising = bull_power_aligned[i] > prev_bull_power[i]
        bear_power_falling = bear_power_aligned[i] < prev_bear_power[i]
        
        # Regime filters
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20  # Exit when trend weakens
        
        if position == 0:  # Flat - look for entry
            # Long: Bull Power > 0 AND rising AND strong trend
            if bull_power_aligned[i] > 0 and bull_power_rising and strong_trend:
                position = 1
                signals[i] = 0.25
            # Short: Bear Power > 0 AND falling AND strong trend
            elif bear_power_aligned[i] > 0 and bear_power_falling and strong_trend:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when Bull Power <= 0 OR trend weakens
            exit_condition = (bull_power_aligned[i] <= 0) or weak_trend
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when Bear Power <= 0 OR trend weakens
            exit_condition = (bear_power_aligned[i] <= 0) or weak_trend
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals