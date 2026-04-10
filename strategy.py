#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX)
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Regime: 1d ADX > 25 = trending (follow Elder Ray), ADX <= 25 = ranging (fade Elder Ray extremes)
# - Long when Bull Power > 0 AND (if trending: Bear Power < 0; if ranging: Bull Power > 0.8*20-bar max Bull Power)
# - Short when Bear Power > 0 AND (if trending: Bull Power < 0; if ranging: Bear Power > 0.8*20-bar max Bear Power)
# - Uses 6h for Elder Ray calculation, 1d for ADX regime filter
# - Elder Ray measures bull/bear strength relative to EMA, effective in both trends and ranges
# - Regime filter prevents whipsaws: in trends follow momentum, in ranges fade extremes
# - Target: 12-25 trades/year to minimize fee drag while capturing regime-appropriate moves

name = "6h_1d_elder_ray_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for ADX regime (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_ema(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        for i in range(len(data)):
            if np.isnan(data[i]):
                if i == 0:
                    result[i] = np.nan
                else:
                    result[i] = result[i-1]
            else:
                if np.isnan(result[i-1]):
                    result[i] = data[i]
                else:
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    atr = wilders_ema(tr, period)
    dmp_smoothed = wilders_ema(dm_plus, period)
    dmm_smoothed = wilders_ema(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dmp_smoothed / atr, 0)
    di_minus = np.where(atr != 0, 100 * dmm_smoothed / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_ema(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Pre-compute 6h Elder Ray components
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Pre-compute 20-bar max for Elder Ray (for regime fading thresholds)
    max_bull_power = pd.Series(bull_power).rolling(window=20, min_periods=1).max().values
    max_bear_power = pd.Series(bear_power).rolling(window=20, min_periods=1).max().values
    
    for i in range(13, n):  # Start after EMA13 warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(max_bull_power[i]) or np.isnan(max_bear_power[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d ADX > 25 = trending, <= 25 = ranging
        is_trending = adx_aligned[i] > 25
        
        # Elder Ray signals
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] > 0
        
        # Regime-specific conditions
        if is_trending:
            # Trending regime: follow Elder Ray momentum
            long_signal = bull_strong and bear_power[i] < 0  # Bull Power +, Bear Power -
            short_signal = bear_strong and bull_power[i] < 0  # Bear Power +, Bull Power -
        else:
            # Ranging regime: fade Elder Ray extremes (require strong readings)
            long_signal = bull_strong and bull_power[i] > 0.8 * max_bull_power[i]
            short_signal = bear_strong and bear_power[i] > 0.8 * max_bear_power[i]
        
        # Exit conditions: opposite Elder Ray signal or power fading
        exit_long = bear_power[i] > 0 or bull_power[i] < 0
        exit_short = bull_power[i] > 0 or bear_power[i] < 0
        
        # Trading logic
        if long_signal:
            if position != 1:  # Only signal on new long entry
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif short_signal:
            if position != -1:  # Only signal on new short entry
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:
            # Check for exits
            if position == 1 and exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short:
                position = 0
                signals[i] = 0.0
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals