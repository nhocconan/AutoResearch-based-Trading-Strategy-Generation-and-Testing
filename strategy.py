#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme reversal with volume confirmation and ATR trailing stop.
# Long when 1d Williams %R crosses above -80 (oversold reversal) with volume > 2.0x median volume.
# Short when 1d Williams %R crosses below -20 (overbought reversal) with volume > 2.0x median volume.
# Uses discrete position size 0.25. Exits when price reaches opposite Camarilla level (S1 for long, R1 for short) or ATR stoploss hits (2.5x ATR).
# Williams %R identifies momentum extremes; reversal with volume filter captures mean reversion in both bull/bear markets.
# 4h timeframe targets 19-50 trades/year (75-200 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # === 1d Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # === 4h Indicators: ATR (14-period) for stoploss ===
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (4h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    # ATR is already on primary timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 20)  # Williams %R, Volume median
    
    # Track position state and entry price for ATR stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_median_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        price = close[i]
        wr = williams_r_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_14[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 2.0x median volume
        volume_spike = current_vol_1d > (vol_median * 2.0)
        
        # Williams %R crossover signals
        wr_cross_up_80 = (wr > -80) and (i == warmup or williams_r_aligned[i-1] <= -80)
        wr_cross_down_20 = (wr < -20) and (i == warmup or williams_r_aligned[i-1] >= -20)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price reaches or falls below S1 (mean reversion to support)
            # OR ATR stoploss hit (2.5 * ATR below entry)
            if price <= s1 or price <= entry_price - 2.5 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price reaches or rises above R1 (mean reversion to resistance)
            # OR ATR stoploss hit (2.5 * ATR above entry)
            if price >= r1 or price >= entry_price + 2.5 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R crosses above -80 (oversold reversal) with volume spike
            if wr_cross_up_80 and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R crosses below -20 (overbought reversal) with volume spike
            elif wr_cross_down_20 and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_WilliamsR_1dVolumeSpike2.0x_CamarillaExit_ATRTrail2.5_v1"
timeframe = "4h"
leverage = 1.0