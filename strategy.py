#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter with 1d trend confirmation
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0, Bear Power improving (less negative), and ADX > 25 (trending)
# - Short when Bear Power < 0, Bull Power deteriorating (less positive), and ADX > 25 (trending)
# - Uses 1d EMA200 for higher timeframe trend filter (only long in 1d uptrend, short in downtrend)
# - Volume confirmation: current volume > 1.5x 20-period average
# - Discrete position sizing (0.25) to minimize fee churn
# - Targets 12-35 trades/year (50-140 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets by adapting to regime (ADX filter) and HTF trend

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # ADX calculation (14-period)
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
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14 = WilderSmooth(tr, 14)
    dm_plus_14 = WilderSmooth(dm_plus, 14)
    dm_minus_14 = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (atr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (atr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = WilderSmooth(dx, 14)
    
    # Volume confirmation: > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend deterioration or HTF trend change
            if (bull_power[i] <= 0 or  # Bull Power turned negative
                bear_power[i] > bear_power[i-1] or  # Bear Power worsening (increasing)
                adx[i] < 20 or  # Trend weakening
                (prices['close'].iloc[i] < ema_200_1d_aligned[i] and ema_200_1d_aligned[i] > ema_200_1d_aligned[i-1])):  # 1d trend turned down
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend deterioration or HTF trend change
            if (bear_power[i] <= 0 or  # Bear Power turned negative
                bull_power[i] < bull_power[i-1] or  # Bull Power worsening (decreasing)
                adx[i] < 20 or  # Trend weakening
                (prices['close'].iloc[i] > ema_200_1d_aligned[i] and ema_200_1d_aligned[i] < ema_200_1d_aligned[i-1])):  # 1d trend turned up
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entry with all conditions aligned
            if vol_spike[i] and adx[i] > 25:  # Volume spike and strong trend
                # Long signal: Bull Power positive, Bear Power improving (less negative), 1d uptrend
                if (bull_power[i] > 0 and 
                    bear_power[i] < bear_power[i-1] and  # Bear Power decreasing (improving)
                    prices['close'].iloc[i] > ema_200_1d_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short signal: Bear Power negative, Bull Power deteriorating (less positive), 1d downtrend
                elif (bear_power[i] > 0 and 
                      bull_power[i] < bull_power[i-1] and  # Bull Power decreasing (worsening)
                      prices['close'].iloc[i] < ema_200_1d_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals