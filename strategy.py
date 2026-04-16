#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Bollinger Band squeeze breakout from 1d timeframe.
# Long when price breaks above upper BB(20,2) with volume > 1.5x 20-period median volume AND BB width at 20-period low (squeeze).
# Short when price breaks below lower BB(20,2) with volume > 1.5x 20-period median volume AND BB width at 20-period low.
# Uses discrete position size 0.25. Exits when price reaches opposite BB band (mean reversion) or ATR stoploss hits (2.0x ATR).
# Bollinger squeeze identifies low volatility periods primed for explosive moves. Volume confirmation filters false breakouts.
# 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. Works in both bull/bear markets as it trades volatility expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Bollinger Bands (20,2) ===
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # === 1d Indicators: BB Width percentile (20-period) for squeeze detection ===
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # === 1d Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # === 6h Indicators: ATR (14-period) for stoploss ===
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (6h)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    # ATR is already on primary timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14)  # BB calculations, ATR
    
    # Track position state and entry price for ATR stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(vol_median_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        price = close[i]
        bb_upper = bb_upper_aligned[i]
        bb_lower = bb_lower_aligned[i]
        bb_width_percentile = bb_width_percentile_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_14[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.5x median volume
        volume_spike = current_vol_1d > (vol_median * 1.5)
        
        # Squeeze filter: BB width at 20-period low (bottom 20% percentile)
        squeeze = bb_width_percentile <= 20
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price reaches or falls below lower BB (mean reversion)
            # OR ATR stoploss hit (2.0 * ATR below entry)
            if price <= bb_lower or price <= entry_price - 2.0 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price reaches or rises above upper BB (mean reversion)
            # OR ATR stoploss hit (2.0 * ATR above entry)
            if price >= bb_upper or price >= entry_price + 2.0 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above upper BB with volume spike AND squeeze
            if price > bb_upper and volume_spike and squeeze:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below lower BB with volume spike AND squeeze
            elif price < bb_lower and volume_spike and squeeze:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_BollingerSqueeze_Breakout_VolumeSpike1.5x_ATRTrail2.0_v1"
timeframe = "6h"
leverage = 1.0