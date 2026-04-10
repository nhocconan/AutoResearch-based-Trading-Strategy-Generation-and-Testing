#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# - Uses Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) on 6h
# - Regime filter from 1d: ADX > 25 for trending (follow Elder Ray), ADX < 20 for ranging (fade Elder Ray extremes)
# - In trending regime: long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
# - In ranging regime: long when Bear Power < -std(Bear Power) (oversold), short when Bull Power < -std(Bull Power) (overbought)
# - Volume confirmation: current 6h volume > 1.5x 20-period average
# - Discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets via regime adaptation

name = "6h_1d_elder_ray_regime_volume_v1"
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
    
    # Pre-compute 1d indicators for regime
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ADX(14) for regime detection
    # True Range
    high_low = high_1d - low_1d
    high_close = np.abs(np.roll(high_1d, 1) - close_1d)
    high_close[0] = np.nan  # First value has no previous close
    low_close = np.abs(np.roll(low_1d, 1) - close_1d)
    low_close[0] = np.nan
    tr = np.nanmax(np.column_stack([high_low, high_close, low_close]), axis=1)
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=np.nan)
    down_move = -np.diff(low_1d, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def _wilder_smooth(x, period):
        result = np.full_like(x, np.nan, dtype=float)
        if len(x) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(x[:period])
            # Wilder smoothing
            for i in range(period, len(x)):
                result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr_1d = _wilder_smooth(tr, 14)
    plus_di_1d = 100 * _wilder_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * _wilder_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = _wilder_smooth(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 6h indicators
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # 6h EMA(13) for Elder Ray
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_6h = high_6h - ema_13_6h  # High - EMA13
    bear_power_6h = ema_13_6h - low_6h   # EMA13 - Low
    
    # Volume confirmation: > 1.5x 20-period average
    avg_volume_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike_6h = volume_6h > (1.5 * avg_volume_20_6h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13_6h[i]) or 
            np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(vol_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        bull_power = bull_power_6h[i]
        bear_power = bear_power_6h[i]
        vol_spike = vol_spike_6h[i]
        
        # Regime determination
        if adx > 25:
            regime = 'trending'
        elif adx < 20:
            regime = 'ranging'
        else:
            regime = 'transition'  # Hold previous signals
        
        if regime == 'transition':
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if regime == 'trending':
            # Follow Elder Ray momentum
            if i > 50:
                bull_power_prev = bull_power_6h[i-1]
                bear_power_prev = bear_power_6h[i-1]
                
                # Long when Bull Power > 0 and rising
                if bull_power > 0 and bull_power > bull_power_prev and vol_spike:
                    if position != 1:
                        position = 1
                        signals[i] = 0.25
                    else:
                        signals[i] = 0.25
                # Short when Bear Power > 0 and rising
                elif bear_power > 0 and bear_power > bear_power_prev and vol_spike:
                    if position != -1:
                        position = -1
                        signals[i] = -0.25
                    else:
                        signals[i] = -0.25
                else:
                    # Hold current position or flat
                    signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            else:
                signals[i] = 0.0
        
        elif regime == 'ranging':
            # Fade Elder Ray extremes (mean reversion)
            # Calculate rolling standard deviations for power metrics
            if i >= 70:  # Need enough data for std calculation
                lookback = 50
                bp_std = np.nanstd(bull_power_6h[max(0, i-lookback):i+1])
                br_std = np.nanstd(bear_power_6h[max(0, i-lookback):i+1])
                
                if not (np.isnan(bp_std) or np.isnan(br_std) or bp_std == 0 or br_std == 0):
                    # Long when Bear Power is deeply negative (oversold)
                    if bear_power < -br_std and vol_spike:
                        if position != 1:
                            position = 1
                            signals[i] = 0.25
                        else:
                            signals[i] = 0.25
                    # Short when Bull Power is deeply negative (overbought)
                    elif bull_power < -bp_std and vol_spike:
                        if position != -1:
                            position = -1
                            signals[i] = -0.25
                        else:
                            signals[i] = -0.25
                    else:
                        # Hold current position or flat
                        signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
                else:
                    signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            else:
                signals[i] = 0.0
    
    return signals