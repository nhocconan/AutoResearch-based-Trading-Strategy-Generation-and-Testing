#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1d Regime Filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Bullish when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending)
# - Bearish when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (trending)
# - Exit when power weakens (Bull Power < 0 for longs, Bear Power < 0 for shorts) OR ADX < 20
# - Uses 6h for Elder Ray calculation and 1d for regime (ADX) filter
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Works in both bull and bear: ADX filter ensures we only trade strong trends,
#   Elder Ray measures bull/bear power behind the move

name = "6h_1d_elder_ray_power_regime_v1"
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
    
    # Pre-compute 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
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
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI and DX
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX (smoothed DX)
    adx_1d = wilders_smoothing(dx, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_6h = prices['close'].values
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema13_6h  # High - EMA
    bear_power = ema13_6h - low_6h   # EMA - Low
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema13_6h[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        adx = adx_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        
        if position == 0:  # Flat - look for new entries
            # Enter long when bull power positive, bear power negative, and strong trend
            if bp > 0 and br > 0 and adx > 25:
                position = 1
                signals[i] = 0.25
            # Enter short when bear power positive, bull power negative, and strong trend
            elif br > 0 and bp > 0 and adx > 25:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            exit_signal = False
            if position == 1:  # Long position
                # Exit when bull power turns negative OR trend weakens
                if bp <= 0 or adx < 20:
                    exit_signal = True
            elif position == -1:  # Short position
                # Exit when bear power turns negative OR trend weakens
                if br <= 0 or adx < 20:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals