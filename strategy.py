#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Regime Filter
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Regime: 1d ADX(14) > 25 = trending (follow Elder Ray), ADX < 20 = ranging (fade Elder Ray extremes)
# - Long when Bull Power > 0 AND Bear Power < 0 AND (trending regime OR (ranging and Bull Power > 0.5*ATR))
# - Short when Bear Power > 0 AND Bull Power < 0 AND (trending regime OR (ranging and Bear Power > 0.5*ATR))
# - Exit when Elder Power reverses sign or opposite Elder Power > 0
# - Position sizing: 0.25 discrete
# - Target: 12-37 trades/year on 6h (50-150 total over 4 years)
# - Works in bull/bear via regime adaptation: trends follow momentum, ranges fade extremes

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Primary 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate ATR(14) for regime thresholds
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # --- Multi-Timeframe: 1d Regime Filter ---
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = pd.Series(high_1d - low_1d)
    tr2_1d = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3_1d = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean()
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean()
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe (completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- Signal Generation ---
    for i in range(13, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime classification
        adx_val = adx_aligned[i]
        if adx_val > 25:
            regime = 'trending'  # Follow Elder Ray momentum
        elif adx_val < 20:
            regime = 'ranging'   # Fade Elder Ray extremes
        else:
            regime = 'transition' # Neutral - require stronger signal
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] > 0
        bull_weak = bull_power[i] <= 0
        bear_weak = bear_power[i] <= 0
        
        # Threshold for ranging regime
        threshold = 0.5 * atr[i]
        
        if position == 0:  # Flat - look for entry
            long_signal = False
            short_signal = False
            
            if regime == 'trending':
                # In trending markets, follow Elder Ray direction
                long_signal = bull_strong and bear_weak
                short_signal = bear_strong and bull_weak
            elif regime == 'ranging':
                # In ranging markets, fade extremes but require stronger signal
                long_signal = bull_power[i] > threshold and bear_power[i] < -threshold
                short_signal = bear_power[i] > threshold and bull_power[i] < -threshold
            else:  # transition
                # Require clear dominance
                long_signal = bull_power[i] > threshold and bear_power[i] < 0
                short_signal = bear_power[i] > threshold and bull_power[i] < 0
            
            if long_signal:
                position = 1
                signals[i] = 0.25
            elif short_signal:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
                
        elif position == 1:  # Long position - look for exit
            # Exit when Bull Power turns negative OR Bear Power becomes positive
            exit_condition = bull_power[i] <= 0 or bear_power[i] > 0
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when Bear Power turns negative OR Bull Power becomes positive
            exit_condition = bear_power[i] <= 0 or bull_power[i] > 0
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals