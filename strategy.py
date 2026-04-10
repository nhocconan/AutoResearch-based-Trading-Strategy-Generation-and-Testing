#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d ADX regime filter and volume confirmation
# - Bull Power = High - EMA(13); Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND 1d ADX > 25 (strong trend) AND 6h volume > 1.5x 20-bar avg
# - Short when Bear Power > 0 AND 1d ADX > 25 (strong trend) AND 6h volume > 1.5x 20-bar avg
# - Exit when Power reverses sign (Bull Power < 0 for longs, Bear Power < 0 for shorts)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Elder Ray measures trend strength via price position relative to EMA
# - 1d ADX filter ensures we only trade in strong trending regimes (avoids chop)
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: ADX filter adapts to trending conditions

name = "6h_1d_elder_power_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(np.abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - pd.Series(high_1d).shift(1)) > (pd.Series(low_1d).shift(1) - low_1d),
                                 np.maximum(high_1d - pd.Series(high_1d).shift(1), 0), 0))
    dm_minus = pd.Series(np.where((pd.Series(low_1d).shift(1) - low_1d) > (high_1d - pd.Series(high_1d).shift(1)),
                                  np.maximum(pd.Series(low_1d).shift(1) - low_1d, 0), 0))
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # ADX > 25 indicates strong trend
    adx_strong = adx > 25
    
    # Align 1d ADX regime to 6h timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong)
    
    # Pre-compute EMA(13) for 6h data (Elder Ray base)
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray Power
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema_13  # High - EMA(13)
    bear_power = ema_13 - low   # EMA(13) - Low
    
    # Power conditions
    bull_power_pos = bull_power > 0
    bear_power_pos = bear_power > 0
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_strong_aligned[i]) or np.isnan(bull_power_pos[i]) or
            np.isnan(bear_power_pos[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power positive AND 1d ADX strong trend AND volume spike
            if (bull_power_pos[i] and 
                adx_strong_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power positive AND 1d ADX strong trend AND volume spike
            elif (bear_power_pos[i] and 
                  adx_strong_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Power reverses
            # Exit when Power reverses sign (Bull Power < 0 for longs, Bear Power < 0 for shorts)
            if position == 1:  # Long position
                exit_signal = bull_power[i] < 0  # Bull Power turned negative
            else:  # Short position
                exit_signal = bear_power[i] < 0  # Bear Power turned negative
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals