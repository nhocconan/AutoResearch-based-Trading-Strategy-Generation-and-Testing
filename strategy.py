#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d volume confirmation + 1w ADX regime filter
# Elder Ray measures bull/bear power relative to EMA13: bull_power = high - EMA, bear_power = low - EMA
# Volume confirmation ensures institutional participation
# 1w ADX regime filter: ADX > 25 = trending (follow Elder Ray signals), ADX < 20 = ranging (fade Elder Ray extremes)
# Works in bull/bear: regime filter adapts, Elder Ray captures momentum exhaustion and continuation
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_1d_1w_elder_ray_volume_adx_v1"
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
    
    # Load 1d data ONCE before loop for EMA and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(13) for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Bull Power and Bear Power (Elder Ray)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d  # negative values indicate bearish pressure
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar close)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Align 1w ADX to 6h timeframe (wait for 1w bar close)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Regime filter: ADX > 25 = trending (follow Elder Ray), ADX < 20 = ranging (fade Elder Ray extremes)
        trending_regime = adx_1w_aligned[i] > 25
        ranging_regime = adx_1w_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit: bear power turns negative OR regime shifts to ranging
            if bear_power_1d_aligned[i] > -0.5 * np.std(bear_power_1d_aligned[max(0, i-50):i+1]) or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bull power turns positive OR regime shifts to ranging
            if bull_power_1d_aligned[i] < 0.5 * np.std(bull_power_1d_aligned[max(0, i-50):i+1]) or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime and volume_confirmed:
                # Follow Elder Ray in trending regime
                if bull_power_1d_aligned[i] > 0 and bear_power_1d_aligned[i] < 0:
                    # Both bulls and bears active - check momentum
                    if bull_power_1d_aligned[i] > abs(bear_power_1d_aligned[i]):
                        position = 1
                        signals[i] = 0.25
                    elif bear_power_1d_aligned[i] < -bull_power_1d_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                elif bull_power_1d_aligned[i] > 0:
                    position = 1
                    signals[i] = 0.25
                elif bear_power_1d_aligned[i] < 0:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime and volume_confirmed:
                # Fade Elder Ray extremes in ranging regime
                if bull_power_1d_aligned[i] > 2 * np.std(bull_power_1d_aligned[max(0, i-50):i+1]):
                    position = -1  # fade extreme bull power
                    signals[i] = -0.25
                elif bear_power_1d_aligned[i] < -2 * np.std(bear_power_1d_aligned[max(0, i-50):i+1]):
                    position = 1  # fade extreme bear power
                    signals[i] = 0.25
    
    return signals