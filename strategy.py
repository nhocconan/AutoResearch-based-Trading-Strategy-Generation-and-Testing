#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX Regime + Volume Spike
# Bull power (high - EMA13) and Bear power (EMA13 - low) with ADX > 25 for trending markets.
# Long when Bull power > 0 AND Bear power < 0 AND ADX > 25 AND volume spike.
# Short when Bull power < 0 AND Bear power > 0 AND ADX > 25 AND volume spike.
# Uses 1d EMA13 for power calculation and 1d ADX for regime filter to avoid whipsaws.
# Volume spike confirms conviction. Discrete sizing 0.25 to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Works in bull (trend + power divergence) and bear (strong trends with volume).
# Timeframe: 6h (primary timeframe as required).

name = "6h_ElderRay_ADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA13 and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Rest: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Higher threshold for fewer trades
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to insufficient data for indicators)
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bull power = high - EMA13, Bear power = EMA13 - low
            bull_power = high[i] - ema_13_1d_aligned[i]
            bear_power = ema_13_1d_aligned[i] - low[i]
            
            # Long: Bull power > 0 AND Bear power > 0 (both positive indicates strong uptrend) 
            #       AND ADX > 25 (trending) AND volume spike
            if (bull_power > 0 and bear_power > 0 and 
                adx_1d_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull power < 0 AND Bear power < 0 (both negative indicates strong downtrend)
            #        AND ADX > 25 (trending) AND volume spike
            elif (bull_power < 0 and bear_power < 0 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX drops below 20 (trend weakening) OR power divergence fades
            bull_power = high[i] - ema_13_1d_aligned[i]
            bear_power = ema_13_1d_aligned[i] - low[i]
            if (adx_1d_aligned[i] < 20 or 
                bull_power <= 0 or bear_power <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX drops below 20 (trend weakening) OR power divergence fades
            bull_power = high[i] - ema_13_1d_aligned[i]
            bear_power = ema_13_1d_aligned[i] - low[i]
            if (adx_1d_aligned[i] < 20 or 
                bull_power >= 0 or bear_power >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals