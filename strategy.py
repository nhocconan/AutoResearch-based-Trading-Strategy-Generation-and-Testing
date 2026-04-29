#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (strong trend)
# Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (strong trend)
# Exit when Elder Power signals weaken OR 1d ADX drops below 20 (weak trend/ranging)
# Uses 13-period EMA for Elder Ray calculation (standard setting)
# ADX filter ensures we only trade in strong trending markets, avoiding whipsaws in ranges
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Elder Ray captures trend strength via price-EMA relationship, ADX confirms trend validity

name = "6h_ElderRay_ADX_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX regime filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter (standard 14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ , DM- (using Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    dm_plus_smooth[0] = dm_plus[0]
    dm_minus_smooth[0] = dm_minus[0]
    for i in range(1, len(dm_plus)):
        dm_plus_smooth[i] = (1 - alpha) * dm_plus_smooth[i-1] + alpha * dm_plus[i]
        dm_minus_smooth[i] = (1 - alpha) * dm_minus_smooth[i-1] + alpha * dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    adx = np.zeros_like(dx)
    adx[period-1] = np.mean(dx[:period])  # First ADX is average of first 'period' DX values
    for i in range(period, len(dx)):
        adx[i] = (1 - alpha) * adx[i-1] + alpha * dx[i]
    
    # Elder Ray calculation: Bull Power = High - EMA(close), Bear Power = Low - EMA(close)
    # Using 13-period EMA as standard
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13  # Bull Power
    bear_power = low_1d - ema_13   # Bear Power
    
    # Align all 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_adx = adx_aligned[i]
        curr_bull = bull_power_aligned[i]
        curr_bear = bear_power_aligned[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Elder Ray signals weaken OR ADX drops below 20 (weak trend)
            if curr_bull <= 0 or curr_bear >= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray signals weaken OR ADX drops below 20 (weak trend)
            if curr_bull >= 0 or curr_bear <= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong uptrend)
            if curr_bull > 0 and curr_bear < 0 and curr_adx > 25:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (strong downtrend)
            elif curr_bear > 0 and curr_bull < 0 and curr_adx > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals