#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with daily regime filter
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Regime filter: 1d ADX > 25 for trending, < 20 for ranging
# In trending (ADX>25): go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
# In ranging (ADX<20): mean reversion at Bollinger Bands (20,2)
# Exit when power crosses zero or opposite signal
# Designed to work in both bull (trending) and bear (ranging) markets

name = "6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: EMA(13) of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = ema_13 - low   # Bear Power = EMA - Low
    
    # 1d ADX for regime filter (trending vs ranging)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 0])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing = alpha = 1/period)
    def wilders_smooth(data, period):
        alpha = 1.0 / period
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    atr_period = 14
    tr_smooth = wilders_smooth(tr, atr_period)
    dm_plus_smooth = wilders_smooth(dm_plus, atr_period)
    dm_minus_smooth = wilders_smooth(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilders_smooth(dx, atr_period)
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Bollinger Bands for ranging regime (20,2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + bb_std * bb_std_dev
    bb_lower = sma_20 - bb_std * bb_std_dev
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_13[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(sma_20[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Bull Power crosses below zero OR Bear Power > Bull Power (momentum shift)
            if bull_power[i] <= 0 or bear_power[i] > bull_power[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bear Power crosses below zero OR Bull Power > Bear Power (momentum shift)
            if bear_power[i] <= 0 or bull_power[i] > bear_power[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Regime-based entry logic
            if adx_aligned[i] > 25:  # Trending regime
                # Long: Bull Power > 0 and rising (current > previous)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power > 0 and rising (current > previous)
                elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
            elif adx_aligned[i] < 20:  # Ranging regime
                # Mean reversion at Bollinger Bands
                # Long: price touches or goes below lower band
                if close[i] <= bb_lower[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price touches or goes above upper band
                elif close[i] >= bb_upper[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals