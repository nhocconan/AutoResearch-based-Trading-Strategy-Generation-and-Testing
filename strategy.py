#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power vs EMA13: bull_power = high - EMA13, bear_power = low - EMA13
# 1d ADX > 25 indicates trending regime (follow Elder Ray signals), ADX < 20 indicates ranging (fade extremes)
# Volume confirmation > 1.5x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~15-30 trades/year per symbol with 0.25 sizing
# Works in bull markets via bull power > 0 in uptrend, bear markets via bear power < 0 in downtrend
# In ranging markets, fades when power reaches extreme deviations from zero

name = "6h_ElderRay_1dADX_Regime_Volume_v1"
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
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value: simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    atr = wilders_smoothing(tr, period_adx)
    dm_plus_smooth = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smooth = wilders_smoothing(dm_minus, period_adx)
    
    # Directional Indicators
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period_adx)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h EMA13 for Elder Ray calculation
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d ADX (30 periods) + 6h EMA13 (13 periods)
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d regime: ADX > 25 = trending, ADX < 20 = ranging
        trending_regime = adx_aligned[i] > 25
        ranging_regime = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            if trending_regime:
                # In trend: follow Elder Ray direction
                if bull_power[i] > 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif ranging_regime:
                # In range: fade extreme power readings
                # Calculate rolling z-score of power for extreme detection
                lookback = min(50, i+1)  # dynamic lookback
                bp_mean = np.nanmean(bull_power[max(0, i-lookback+1):i+1])
                bp_std = np.nanstd(bull_power[max(0, i-lookback+1):i+1])
                br_mean = np.nanmean(bear_power[max(0, i-lookback+1):i+1])
                br_std = np.nanstd(bear_power[max(0, i-lookback+1):i+1])
                
                bp_z = (bull_power[i] - bp_mean) / bp_std if bp_std > 0 else 0
                br_z = (bear_power[i] - br_mean) / br_std if br_std > 0 else 0
                
                # Fade when power reaches extreme levels (2+ std dev)
                if bp_z > 2.0 and volume_spike[i]:  # Extreme bull power -> short
                    signals[i] = -0.25
                    position = -1
                elif br_z < -2.0 and volume_spike[i]:  # Extreme bear power -> long
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX between 20-25) - wait for clearer signal
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if trending_regime:
                # In trend: exit when bull power turns negative
                if bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # In range: exit when power normalizes
                lookback = min(50, i+1)
                bp_mean = np.nanmean(bull_power[max(0, i-lookback+1):i+1])
                bp_std = np.nanstd(bull_power[max(0, i-lookback+1):i+1])
                bp_z = (bull_power[i] - bp_mean) / bp_std if bp_std > 0 else 0
                if bp_z < 0.5:  # Power returned to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            if trending_regime:
                # In trend: exit when bear power turns positive
                if bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # In range: exit when power normalizes
                lookback = min(50, i+1)
                br_mean = np.nanmean(bear_power[max(0, i-lookback+1):i+1])
                br_std = np.nanstd(bear_power[max(0, i-lookback+1):i+1])
                br_z = (bear_power[i] - br_mean) / br_std if br_std > 0 else 0
                if br_z > -0.5:  # Power returned to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals