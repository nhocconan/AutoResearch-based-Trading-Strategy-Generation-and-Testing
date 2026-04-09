#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter
# - Elder Ray: Bull Power = high - EMA(13), Bear Power = low - EMA(13) on 6h
# - 1d ADX > 25 indicates trending regime (use Elder Ray for direction)
# - 1d ADX < 20 indicates ranging regime (fade extreme Elder Ray values)
# - In trend: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
# - In range: long when Bear Power < -0.5*ATR and turning up, short when Bull Power > 0.5*ATR and turning down
# - Volume confirmation: 6h volume > 1.5 * 20-period average
# - Target: 12-30 trades/year on 6h (50-120 total over 4 years) to minimize fee drag
# - Works in bull via trend continuation, in bear via trend continuation or range fading

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 6h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = high - EMA
    bear_power = low - ema_13   # Bear Power = low - EMA
    
    # 6h ATR(14) for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_ok = volume_confirm[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when bull power turns negative
                if bull <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit when bear power rises above -0.2*ATR (mean reversion)
                if bear > -0.2 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when bear power turns positive
                if bear >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit when bull power falls below 0.2*ATR (mean reversion)
                if bull < 0.2 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry conditions
            if vol_ok:
                if adx > 25:  # Trending regime - follow Elder Ray momentum
                    # Long when bull power positive and rising
                    if bull > 0 and i > 100 and bull > bull_power[i-1]:
                        position = 1
                        signals[i] = 0.25
                    # Short when bear power negative and falling
                    elif bear < 0 and i > 100 and bear < bear_power[i-1]:
                        position = -1
                        signals[i] = -0.25
                else:  # Ranging regime - fade extreme Elder Ray values
                    # Long when bear power is deeply negative and turning up
                    if bear < -0.5 * atr[i] and i > 100 and bear > bear_power[i-1]:
                        position = 1
                        signals[i] = 0.25
                    # Short when bull power is deeply positive and turning down
                    elif bull > 0.5 * atr[i] and i > 100 and bull < bull_power[i-1]:
                        position = -1
                        signals[i] = -0.25
    
    return signals