#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R reversal with volume confirmation and ATR trailing stop.
# Long when 1d Williams %R crosses above -80 (oversold reversal) with volume > 1.5x median volume.
# Short when 1d Williams %R crosses below -20 (overbought reversal) with volume > 1.5x median volume.
# Uses discrete position size 0.25. Exits when price reaches opposite Williams %R level (-50 for long, -50 for short) or ATR stoploss hits (2.0x ATR).
# Williams %R identifies momentum extremes; reversal with volume filter captures mean reversion in both bull/bear markets.
# 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Williams %R (14-period) ===
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    williams_r = -100 * ((highest_high - close_1d) / (highest_high - lowest_low))
    williams_r_values = williams_r.values
    
    # === 1d Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # === 12h Indicators: ATR (14-period) for stoploss ===
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (12h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_values)
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
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_median_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        price = close[i]
        wr_val = williams_r_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_14[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.5x median volume
        volume_spike = current_vol_1d > (vol_median * 1.5)
        
        # Williams %R crossover signals
        wr_cross_up_80 = (wr_val > -80) and (i == warmup or williams_r_aligned[i-1] <= -80)
        wr_cross_down_20 = (wr_val < -20) and (i == warmup or williams_r_aligned[i-1] >= -20)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Williams %R crosses above -50 (momentum weakening) OR ATR stoploss hit (2.0 * ATR below entry)
            if wr_val > -50 or price <= entry_price - 2.0 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Williams %R crosses below -50 (momentum weakening) OR ATR stoploss hit (2.0 * ATR above entry)
            if wr_val < -50 or price >= entry_price + 2.0 * atr:
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

name = "12h_WilliamsR14_1dVolumeSpike1.5x_ExitWR-50_ATRTrail2.0_v1"
timeframe = "12h"
leverage = 1.0