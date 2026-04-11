#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume confirmation
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Regime: 1d ADX > 25 = trending (use Elder Ray signals), ADX <= 25 = ranging (fade extremes)
# - In trending regime: Long when Bull Power > 0 and rising, Short when Bear Power > 0 and rising
# - In ranging regime: Long when Bear Power < -0.5 * ATR(10) and turning up, Short when Bull Power < -0.5 * ATR(10) and turning down
# - Volume confirmation: current volume > 1.2 * 20-period average
# - Target: 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits
# - Works in both bull and bear markets by adapting to regime (trending vs ranging)

name = "6h_1d_elder_ray_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for regime filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d indicators for regime detection
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(13) for Elder Ray calculation
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d ADX for regime detection (trending vs ranging)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(tr_14 > 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 > 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 6h Elder Power components
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13_6h  # High - EMA(13)
    bear_power = ema_13_6h - low   # EMA(13) - Low
    
    # Pre-compute 6h ATR for ranging regime thresholds
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_10_6h = pd.Series(tr_6h).rolling(window=10, min_periods=10).mean().values
    
    # Pre-compute 6h volume confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_10_6h[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        bull = bull_power[i]
        bear = bear_power[i]
        volume_current = volume[i]
        atr_10 = atr_10_6h[i]
        
        # Volume confirmation: current volume > 1.2 * 20-period average
        vol_confirm = volume_current > 1.2 * volume_sma_20[i]
        
        # Regime filter: 1d ADX > 25 = trending, ADX <= 25 = ranging
        adx_val = adx_1d_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val <= 25
        
        # Elder Ray momentum (rate of change)
        bull_momentum = bull - (bull_power[i-1] if i > 0 else bull)
        bear_momentum = bear - (bear_power[i-1] if i > 0 else bear)
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        if is_trending and vol_confirm:
            # Trending regime: follow Elder Ray momentum
            if bull > 0 and bull_momentum > 0:
                enter_long = True
            if bear > 0 and bear_momentum > 0:
                enter_short = True
        elif is_ranging and vol_confirm:
            # Ranging regime: fade Elder Ray extremes
            if bear < (-0.5 * atr_10) and bear_momentum > 0:  # Bear power extremely negative and turning up
                enter_long = True
            if bull < (-0.5 * atr_10) and bull_momentum < 0:  # Bull power extremely negative and turning down
                enter_short = True
        
        # Exit conditions: reverse signal or volatility expansion
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bear power becomes positive (bulls losing control) or volatility expands
            exit_long = bear > 0 or (adx_val > 40 and bull_momentum < 0)
        elif position == -1:
            # Exit short if bull power becomes positive (bears losing control) or volatility expands
            exit_short = bull > 0 or (adx_val > 40 and bear_momentum < 0)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals