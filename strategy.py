#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1w/1d regime filter
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Strong trend when Bull Power > 0 and Bear Power < 0 (both bulls and bears in control)
# - Regime filter: Only trade when 1w ADX > 25 (trending market) to avoid chop
# - Entry: Go long when Bull Power crosses above 0 with regime confirmation
# - Entry: Go short when Bear Power crosses above 0 with regime confirmation (Bear Power rising = weakening bears)
# - Exit: Opposite cross or regime breakdown (ADX < 20)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) for 6f
# - Works in bull markets (strong bull power) and bear markets (strong bear power)
# - Weekly ADX ensures we only trade in strong trends, avoiding whipsaws in ranging markets

name = "6h_1w_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 13-period EMA for Elder Ray (using daily close as proxy)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    close_1d = df_1d['close'].values
    # Calculate EMA13 on daily closes
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align daily EMA13 to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Elder Ray components
    bull_power = high - ema13_aligned  # High - EMA13
    bear_power = ema13_aligned - low   # EMA13 - Low
    
    # Calculate weekly ADX for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w).shift(1) - pd.Series(low_1w).shift(1)
    tr2 = abs(pd.Series(high_1w).shift(1) - pd.Series(close_1w).shift(1))
    tr3 = abs(pd.Series(low_1w).shift(1) - pd.Series(close_1w).shift(1))
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = pd.Series(high_1w).diff()
    minus_dm = pd.Series(low_1w).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align weekly ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_14)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Previous values for crossover detection
        bull_power_prev = bull_power[i-1]
        bear_power_prev = bear_power[i-1]
        
        # Regime filter: only trade in trending markets (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20  # Exit regime
        
        # Elder Ray signals
        bull_crossover = (bull_power_prev <= 0) and (bull_power[i] > 0)  # Bull power crosses above zero
        bear_crossover = (bear_power_prev <= 0) and (bear_power[i] > 0)  # Bear power crosses above zero (weakening bears)
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull power crosses above zero in strong trend
        if bull_crossover and strong_trend:
            enter_long = True
        
        # Short: Bear power crosses above zero (indicating weakening bearish momentum) in strong trend
        # This suggests bears are losing control, potential for trend continuation or reversal
        if bear_crossover and strong_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bear power crosses above zero OR trend weakens
            exit_long = bear_crossover or weak_trend
        elif position == -1:
            # Exit short if bull power crosses above zero OR trend weakens
            exit_short = bull_crossover or weak_trend
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals