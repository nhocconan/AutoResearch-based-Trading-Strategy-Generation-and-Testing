#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h EMA50 trend filter + ADX regime
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, 12h EMA50 uptrend, ADX > 25
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, 12h EMA50 downtrend, ADX > 25
# Exit when Elder Ray signals reverse or ADX < 20 (regime change)
# Uses discrete position sizing (0.25) to balance capture and risk.
# Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag.
# Uses 12h for trend and regime, 6h only for signal generation.

name = "6h_ElderRay_ADX_Regime_12hEMA50_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for EMA50 trend and ADX regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h ADX for regime filter (ADX > 25 = trending, < 20 = ranging)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_adx = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h_adx[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h_adx[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smoothed = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smoothed = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di_12h = 100 * dm_plus_smoothed / atr_12h
    minus_di_12h = 100 * dm_minus_smoothed / atr_12h
    
    # Directional Index (DX) and ADX
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx_12h).ewm(alpha=1/14, adjust=False).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Get 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate 6h EMA13 for Elder Ray
    close_6h = df_6h['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_13_6h)
    
    # Calculate Elder Ray components
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    bull_power_6h = high_6h - ema_13_6h
    bear_power_6h = low_6h - ema_13_6h
    
    # Align Elder Ray components to 6h timeframe
    bull_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    ema_13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_13_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # 12h EMA50 and 6h EMA13 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(bull_power_6h_aligned[i]) or np.isnan(bear_power_6h_aligned[i]) or
            np.isnan(ema_13_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bull_power = bull_power_6h_aligned[i]
        curr_bear_power = bear_power_6h_aligned[i]
        curr_ema13_6h = ema_13_6h_aligned[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_adx_12h = adx_12h_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Elder Ray signals reverse or ADX < 20 (regime change to ranging)
            if curr_bull_power <= 0 or curr_adx_12h < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray signals reverse or ADX < 20 (regime change to ranging)
            if curr_bear_power >= 0 or curr_adx_12h < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Regime filter: ADX > 25 (trending market)
            if curr_adx_12h > 25:
                # Trend filter: 12h EMA50 direction
                if curr_ema50_12h > close_12h_adx[-1] if len(close_12h_adx) > 0 else curr_ema50_12h:  # Simplified trend check
                    # Long when Bull Power > 0 and rising, Bear Power < 0
                    if curr_bull_power > 0 and curr_bear_power < 0:
                        signals[i] = 0.25
                        position = 1
                else:
                    # Short when Bear Power < 0 and falling, Bull Power > 0
                    if curr_bear_power < 0 and curr_bull_power > 0:
                        signals[i] = -0.25
                        position = -1
            signals[i] = 0.0
    
    return signals