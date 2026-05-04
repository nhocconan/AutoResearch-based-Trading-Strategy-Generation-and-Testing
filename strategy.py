#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# Uses 6h Elder Ray (Bull/Bear Power) to measure buying/selling pressure
# Filters with 1d ADX regime: ADX>25 = trend (follow Elder Ray), ADX<20 = range (fade Elder Ray extremes)
# Volume confirmation (>1.5x 20 EMA) ensures participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# Works in both bull and bear: regime filter adapts strategy to market conditions.

name = "6h_ElderRay_1dADX_Regime_Volume"
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
    
    # Get 1d data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_period = wilders_smoothing(tr, period)
    dm_plus_period = wilders_smoothing(dm_plus, period)
    dm_minus_period = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_period > 0, (dm_plus_period / tr_period) * 100, 0)
    di_minus = np.where(tr_period > 0, (dm_minus_period / tr_period) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    # Align 1d ADX to 6h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entries
            if adx_aligned[i] > 25:  # Trending regime
                # Follow Elder Ray: long when bull power strong, short when bear power strong
                if bull_power[i] > 0 and bull_power[i] > np.nanmean(bull_power[max(0,i-50):i]) and volume[i] > (1.5 * vol_ema_20[i]):
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and bear_power[i] < np.nanmean(bear_power[max(0,i-50):i]) and volume[i] > (1.5 * vol_ema_20[i]):
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging regime (ADX < 25)
                # Fade Elder Ray extremes: short when bull power extreme, long when bear power extreme
                if bull_power[i] > np.nanpercentile(bull_power[max(0,i-100):i], 85) and volume[i] > (1.5 * vol_ema_20[i]):
                    signals[i] = -0.25  # fade extreme bull power
                    position = -1
                elif bear_power[i] < np.nanpercentile(bear_power[max(0,i-100):i], 15) and volume[i] > (1.5 * vol_ema_20[i]):
                    signals[i] = 0.25   # fade extreme bear power
                    position = 1
        elif position == 1:
            # Exit long: Elder Ray weakness OR regime change to ranging AND power fading
            if (bull_power[i] < 0 or 
                (adx_aligned[i] < 20 and bull_power[i] < np.nanmean(bull_power[max(0,i-30):i]))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Elder Ray weakness OR regime change to ranging AND power fading
            if (bear_power[i] > 0 or 
                (adx_aligned[i] < 20 and bear_power[i] > np.nanmean(bear_power[max(0,i-30):i]))):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals