#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Regime Filter (ADX)
# Elder Ray measures bull/bear power as price deviation from EMA13.
# In trending regimes (ADX > 25), trade with the trend: long when Bull Power > 0 and rising,
# short when Bear Power < 0 and falling. In ranging regimes (ADX <= 25), fade extremes:
# long when Bear Power < -std and rising, short when Bull Power > std and falling.
# Uses 6h timeframe for entries, 1d for regime and EMA13.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Discrete position sizing (0.25) balances return and turnover costs.
# Works in bull/bear via regime adaptation: trend follow in trends, mean revert in ranges.

name = "6h_ElderRay_1dADX_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA13 and ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe (completed 1d bar only)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_aligned
    bear_power = low - ema_13_aligned
    
    # Signals array
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Ensure sufficient history for ADX and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime: ADX > 25 = trending, ADX <= 25 = ranging
        is_trending = adx_aligned[i] > 25
        
        if is_trending:
            # Trend following: trade with the trend
            # Long when Bull Power > 0 and rising (vs previous bar)
            # Short when Bear Power < 0 and falling (vs previous bar)
            if i >= 1:
                bull_rising = bull_power[i] > bull_power[i-1]
                bear_falling = bear_power[i] < bear_power[i-1]
                
                long_signal = bull_power[i] > 0 and bull_rising
                short_signal = bear_power[i] < 0 and bear_falling
            else:
                long_signal = False
                short_signal = False
            
            # Exit on power weakening
            exit_long = bull_power[i] <= 0 or (i >= 1 and bull_power[i] < bull_power[i-1])
            exit_short = bear_power[i] >= 0 or (i >= 1 and bear_power[i] > bear_power[i-1])
        else:
            # Ranging regime: mean reversion at extremes
            # Calculate rolling std of power for dynamic thresholds
            lookback = min(20, i+1)
            if lookback >= 5:
                bull_std = np.std(bull_power[max(0, i-lookback+1):i+1])
                bear_std = np.std(bear_power[max(0, i-lookback+1):i+1])
            else:
                bull_std = bear_std = 1.0  # Fallback
            
            # Long when Bear Power < -std and rising (oversold bounce)
            # Short when Bull Power > std and falling (overbought rejection)
            if i >= 1:
                bear_rising = bear_power[i] > bear_power[i-1]
                bull_falling = bull_power[i] < bull_power[i-1]
                
                long_signal = bear_power[i] < -0.5 * bear_std and bear_rising
                short_signal = bull_power[i] > 0.5 * bull_std and bull_falling
            else:
                long_signal = False
                short_signal = False
            
            # Exit when power returns toward zero
            exit_long = bear_power[i] >= -0.2 * bear_std or (i >= 1 and bear_power[i] < bear_power[i-1])
            exit_short = bull_power[i] <= 0.2 * bull_std or (i >= 1 and bull_power[i] > bear_power[i-1])
        
        # Handle entries and exits
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals