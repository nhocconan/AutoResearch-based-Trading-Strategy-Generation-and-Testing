#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Regime: ADX(14) from 1d - ADX > 25 = trending, ADX < 20 = ranging
# - Entry: Long when Bull Power > 0 AND ADX > 25 (strong uptrend)
#          Short when Bear Power > 0 AND ADX > 25 (strong downtrend)
# - Exit: Opposite signal or ADX < 20 (regime change to ranging)
# - Position sizing: 0.25 discrete levels
# - Uses 1d for regime/trend filter, 6h for Elder Ray calculation
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within HARD MAX: 300 total

name = "6h_1d_elder_ray_regime_v1"
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
    
    # Pre-compute 6h OHLC
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d OHLC for Elder Ray and ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray (1d)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components (1d)
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA(13)
    bear_power_1d = ema_13_1d - low_1d   # Bear Power = EMA(13) - Low
    
    # Calculate ADX(14) from 1d
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
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
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(atr_1d == 0, 1, atr_1d)
    di_minus = 100 * dm_minus_smooth / np.where(atr_1d == 0, 1, atr_1d)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all HTF data to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND ADX > 25 (strong uptrend)
            if bull_power_aligned[i] > 0 and adx_aligned[i] > 25:
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power > 0 AND ADX > 25 (strong downtrend)
            elif bear_power_aligned[i] > 0 and adx_aligned[i] > 25:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Opposite signal OR ADX < 20 (regime change to ranging)
            if position == 1:  # Long position
                if bear_power_aligned[i] > 0 or adx_aligned[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if bull_power_aligned[i] > 0 or adx_aligned[i] < 20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals