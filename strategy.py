#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime with 1d trend filter
# - Uses 1d EMA(50) as primary trend filter (bull if price > EMA50, bear if price < EMA50)
# - In bull regime: long when Elder Ray Bull Power > 0 and ADX(14) > 20
# - In bear regime: short when Elder Ray Bear Power < 0 and ADX(14) > 20
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 6h (50-100 total over 4 years)
# - Combines trend strength (ADX) with momentum (Elder Ray) for high-conviction entries
# - Works in bull markets via Bull Power + ADX, in bear via Bear Power + ADX

name = "6h_1d_elder_ray_adx_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend regime
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # ADX(14) for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(atr != 0, atr, 1e-10)
    di_minus = 100 * dm_minus_smooth / np.where(atr != 0, atr, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx[i]) or adx[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Determine regime: bull if price > 1d EMA50, bear if price < 1d EMA50
        is_bull_regime = close[i] > ema_50_1d_aligned[i]
        is_bear_regime = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: regime change or loss of momentum
            if not is_bull_regime or adx[i] < 20 or bull_power[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: regime change or loss of momentum
            if not is_bear_regime or adx[i] < 20 or bear_power[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for regime-aligned entries with momentum confirmation
            if is_bull_regime and adx[i] > 20 and bull_power[i] > 0:
                position = 1
                signals[i] = 0.25
            elif is_bear_regime and adx[i] > 20 and bear_power[i] < 0:
                position = -1
                signals[i] = -0.25
    
    return signals