#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter + volume confirmation
# - Primary signal: Elder Ray Bull Power (high - EMA13) and Bear Power (low - EMA13) on 6h
#   Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling
# - Regime filter: 12h ADX > 25 (trending market) to avoid whipsaws in ranges
# - Volume confirmation: 6h volume > 1.5x 20-period EMA volume (institutional participation)
# - Position size: 0.25 (discrete level) to balance return and drawdown
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Elder Ray captures momentum strength, ADX filter ensures trending conditions,
#   volume confirmation avoids low-quality breakouts

name = "6h_12h_elderray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h ADX(14) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI and ADX
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_6h = prices['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute Elder Ray components on 6h
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13_6h  # Bull Power = High - EMA13
    bear_power = low_6h - ema_13_6h   # Bear Power = Low - EMA13
    
    # 6h volume regime: volume > 1.5x 20-period EMA volume
    volume = prices['volume'].values
    volume_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_regime = volume > (1.5 * volume_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power becomes positive (momentum shift) OR ADX < 20 (trend weakening) OR volume drops
            if (bear_power[i] > 0 or 
                adx_aligned[i] < 20 or 
                not volume_regime[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power becomes negative (momentum shift) OR ADX < 20 (trend weakening) OR volume drops
            if (bull_power[i] < 0 or 
                adx_aligned[i] < 20 or 
                not volume_regime[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray extremes with ADX filter and volume confirmation
            # Long: Bull Power > 0 AND rising AND ADX > 25 AND volume regime
            # Short: Bear Power < 0 AND falling AND ADX > 25 AND volume regime
            if (bull_power[i] > 0 and 
                bull_power[i] > bull_power[i-1] and  # Rising bull power
                adx_aligned[i] > 25 and 
                volume_regime[i]):
                position = 1
                signals[i] = 0.25
            elif (bear_power[i] < 0 and 
                  bear_power[i] < bear_power[i-1] and  # Falling bear power (more negative)
                  adx_aligned[i] > 25 and 
                  volume_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals