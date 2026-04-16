#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w ATR expansion + volume confirmation + KAMA trend filter.
# Long when 1w ATR > 1.2x 20-period median ATR (expansion) AND price > KAMA(10,2,30) AND volume > 1.5x median volume.
# Short when 1w ATR > 1.2x 20-period median ATR AND price < KAMA(10,2,30) AND volume > 1.5x median volume.
# Uses discrete position size 0.25. Exits when ATR contraction (ATR < 0.8x median ATR) or opposite KAMA cross.
# ATR expansion indicates increased volatility and institutional participation; KAMA filters noise and adapts to market conditions.
# 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
# Weekly timeframe reduces noise and avoids overtrading vs lower timeframes.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for ATR expansion and KAMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # === 1w Indicators: ATR (14-period) ===
    high_low_1w = high_1w - low_1w
    high_close_1w = np.abs(high_1w - np.roll(close_1w, 1))
    low_close_1w = np.abs(low_1w - np.roll(close_1w, 1))
    true_range_1w = np.maximum(high_low_1w, np.maximum(high_close_1w, low_close_1w))
    atr_14_1w = pd.Series(true_range_1w).rolling(window=14, min_periods=14).mean().values
    
    # === 1w Indicators: ATR Median (20-period) for expansion filter ===
    atr_median_20 = pd.Series(atr_14_1w).rolling(window=20, min_periods=20).median().values
    
    # === 1w Indicators: KAMA (10,2,30) ===
    # Efficiency Ratio (ER)
    change_1w = np.abs(np.diff(close_1w, n=1))
    change_1w = np.insert(change_1w, 0, np.nan)
    volatility_1w = np.abs(np.diff(close_1w, n=1))
    volatility_1w = np.insert(volatility_1w, 0, np.nan)
    er_1w = np.zeros_like(close_1w, dtype=float)
    er_1w[:] = np.nan
    sum_change = pd.Series(change_1w).rolling(window=10, min_periods=10).sum().values
    sum_volatility = pd.Series(volatility_1w).rolling(window=10, min_periods=10).sum().values
    er_1w = np.where(sum_volatility != 0, sum_change / sum_volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc_1w = er_1w * (fast_sc - slow_sc) + slow_sc
    sc_1w = sc_1w * sc_1w
    # KAMA calculation
    kama_1w = np.zeros_like(close_1w, dtype=float)
    kama_1w[:] = np.nan
    kama_1w[9] = close_1w[9]  # seed
    for i in range(10, len(close_1w)):
        if not np.isnan(sc_1w[i]):
            kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    
    # === 1w Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (12h)
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    atr_median_aligned = align_htf_to_ltf(prices, df_1w, atr_median_20)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    vol_median_aligned = align_htf_to_ltf(prices, df_1w, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 20, 10)  # ATR needs 14, ATR median needs 20, KAMA needs 10
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_median_aligned[i]) or
            np.isnan(kama_aligned[i]) or np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        atr = atr_14_aligned[i]
        atr_median = atr_median_aligned[i]
        kama = kama_aligned[i]
        vol_median = vol_median_aligned[i]
        
        # Get current 1w volume for volume spike filter
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        current_vol_1w = vol_1w_aligned[i]
        
        # ATR expansion filter: current ATR > 1.2x median ATR
        atr_expansion = atr > (atr_median * 1.2)
        
        # Volume spike filter: current 1w volume > 1.5x median volume
        volume_spike = current_vol_1w > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when ATR contraction OR price < KAMA
            if (atr < atr_median * 0.8) or (price < kama):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when ATR contraction OR price > KAMA
            if (atr < atr_median * 0.8) or (price > kama):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: ATR expansion + price > KAMA + volume spike
            if atr_expansion and (price > kama) and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: ATR expansion + price < KAMA + volume spike
            elif atr_expansion and (price < kama) and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1wATRExpansion_KAMA10_2_30_VolumeSpike1.5x_EXITcontraction_KAMAcross_v1"
timeframe = "12h"
leverage = 1.0