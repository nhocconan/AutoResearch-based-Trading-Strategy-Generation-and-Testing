#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d ADX > 25 indicates trending market (use Elder Ray signals), ADX < 20 indicates ranging (fade extremes)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) for 6h timeframe
# Works in bull markets via trend continuation and bear markets via regime-adaptive mean reversion

name = "6h_ElderRay_1dADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and volume MA)
    start_idx = 30  # buffer for 30-period calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Trending regime (ADX > 25): follow Elder Ray signals
            if adx_aligned[i] > 25:
                # Long: Bull Power > 0 (bulls in control) + volume spike
                if bull_power[i] > 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 (bears in control) + volume spike
                elif bear_power[i] < 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime (ADX < 20): fade Elder Ray extremes
            elif adx_aligned[i] < 20:
                # Long: Bear Power < -std(bear_power) (extreme bearish) + volume spike
                # Short: Bull Power > std(bull_power) (extreme bullish) + volume spike
                # Use rolling std for dynamic thresholds
                lookback = min(i, 50)
                if lookback >= 20:
                    bp_std = np.nanstd(bear_power[max(0, i-lookback):i])
                    bp_std_long = np.nanstd(bull_power[max(0, i-lookback):i])
                    if not (np.isnan(bp_std) or np.isnan(bp_std_long)):
                        if bear_power[i] < -bp_std and volume_spike[i]:
                            signals[i] = 0.25
                            position = 1
                        elif bull_power[i] > bp_std_long and volume_spike[i]:
                            signals[i] = -0.25
                            position = -1
        elif position == 1:  # Long position
            # Exit: Bear Power < 0 (bears take over) or ADX drops below 20 (trend weakening)
            if bear_power[i] < 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Bull Power > 0 (bulls take over) or ADX drops below 20 (trend weakening)
            if bull_power[i] > 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals