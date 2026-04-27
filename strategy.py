#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    Hypothesis: 6h Elder Ray with weekly regime filter
    - Elder Ray (Bull/Bear Power) from daily data captures institutional buying/selling pressure
    - Weekly trend filter (price vs weekly EMA200) ensures trades align with higher timeframe bias
    - Volume confirmation filters weak moves
    - Works in bull/bear by only taking longs in bullish regime, shorts in bearish regime
    - Target: 50-150 total trades over 4 years
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 from daily close for regime filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Elder Ray components from daily data
    # Bull Power = Daily High - EMA13
    # Bear Power = Daily Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 20-period average volume for spike detection on 6h data
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period: need EMA200 (200 days), EMA13 (13 days), and volume MA (20)
    start_idx = max(200, 13, vol_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1d_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Regime filter: bullish if price > weekly EMA200, bearish if price < weekly EMA200
        bullish_regime = price > ema200_1d_aligned[i]
        bearish_regime = price < ema200_1d_aligned[i]
        
        # Elder Ray signals: strong bullish/bearish pressure
        strong_bull = bull_power_aligned[i] > 0 and bull_power_aligned[i] > np.nanmean(bull_power_aligned[max(0, i-20):i])
        strong_bear = bear_power_aligned[i] < 0 and bear_power_aligned[i] < np.nanmean(bear_power_aligned[max(0, i-20):i])
        
        # Volume confirmation: spike > 1.5x average
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long only in bullish regime with strong bull power and volume
            if bullish_regime and strong_bull and volume_confirmation:
                signals[i] = size
                position = 1
            # Short only in bearish regime with strong bear power and volume
            elif bearish_regime and strong_bear and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: bull power turns negative or regime turns bearish
            if bull_power_aligned[i] <= 0 or bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: bear power turns positive or regime turns bullish
            if bear_power_aligned[i] >= 0 or bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_WeeklyEMA200_Regime_Volume"
timeframe = "6h"
leverage = 1.0