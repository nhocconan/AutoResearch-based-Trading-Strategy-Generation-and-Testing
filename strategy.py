#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX) and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Regime filter: ADX(14) > 25 for trending, < 20 for ranging (hysteresis)
# In trending regime (ADX > 25): follow Elder Ray signals (long if Bull Power > 0, short if Bear Power < 0)
# In ranging regime (ADX < 20): mean revert at extremes (long if Bear Power < -std, short if Bull Power > +std)
# Volume confirmation: current volume > 1.5 * 20-period average volume
# Designed for low frequency (50-150 trades over 4 years) to minimize fee drag
# Works in bull/bear via regime adaptation: trend follow in trends, mean revert in ranges

name = "6h_ElderRay_1dADX_Regime_VolumeSpike_v1"
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
    
    # 1d HTF data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d != 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate EMA13 on 6h data for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(30, 20, 13)  # Need ADX(14) with smoothing, volume MA20, EMA13
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime definition with hysteresis
        # Trending: ADX > 25
        # Ranging: ADX < 20
        # Transition: 20 <= ADX <= 25 (hold previous regime)
        
        if i == start_idx:
            # Initialize regime based on current ADX
            if adx > 25:
                regime = 'trending'
            elif adx < 20:
                regime = 'ranging'
            else:
                regime = 'transition'
        else:
            # Propagate previous regime with hysteresis
            prev_regime = regime
            if adx > 25:
                regime = 'trending'
            elif adx < 20:
                regime = 'ranging'
            else:
                regime = prev_regime  # hold in transition zone
        
        if position == 0:  # Flat - look for new entries
            if regime == 'trending':
                # In trending regime: follow Elder Ray signals
                if bull_power[i] > 0 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and vol_spike:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif regime == 'ranging':
                # In ranging regime: mean revert at extremes
                # Calculate volatility-based thresholds
                if i >= 20:
                    # Use recent 20-bar std of Elder Ray power
                    bp_std = np.nanstd(bull_power[max(0, i-20):i+1])
                    br_std = np.nanstd(-bear_power[max(0, i-20):i+1])  # Bear power is negative, so -bear_power for magnitude
                    threshold = max(bp_std, br_std) * 1.5  # 1.5 standard deviations
                    
                    if bear_power[i] < -threshold and vol_spike:  # Strong bear power = oversold
                        signals[i] = 0.25
                        position = 1
                    elif bull_power[i] > threshold and vol_spike:  # Strong bull power = overbought
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            else:  # transition regime - stay flat
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if regime == 'trending':
                # Exit when bull power turns negative (trend weakness)
                if bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif regime == 'ranging':
                # Exit when mean reversion occurs (power returns to neutral)
                if i >= 20:
                    bp_std = np.nanstd(bull_power[max(0, i-20):i+1])
                    if bull_power[i] < bp_std * 0.5:  # Return to half std deviation
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:
                    signals[i] = 0.25
            else:  # transition
                signals[i] = 0.0
                position = 0
        
        elif position == -1:  # Short position
            # Exit conditions
            if regime == 'trending':
                # Exit when bear power turns positive (trend weakness)
                if bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif regime == 'ranging':
                # Exit when mean reversion occurs
                if i >= 20:
                    br_std = np.nanstd(-bear_power[max(0, i-20):i+1])
                    if -bear_power[i] < br_std * 0.5:  # Return to half std deviation
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:  # transition
                signals[i] = 0.0
                position = 0
    
    return signals