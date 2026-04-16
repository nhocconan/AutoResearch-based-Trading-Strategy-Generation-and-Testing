#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with volume confirmation and ATR trailing stop.
# Long when price breaks above R4 with volume > 1.3x 20-period median volume.
# Short when price breaks below S4 with volume > 1.3x 20-period median volume.
# Uses discrete position size 0.25. Exits when price reaches R3/S3 (mean reversion) or ATR stoploss hits (2.5x ATR).
# Camarilla pivots from 12h provide intraday support/resistance levels that work in both trending and ranging markets.
# Volume confirmation filters false breakouts. 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (based on previous 12h bar) ===
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    # Using previous bar's values to avoid look-ahead
    high_prev = np.roll(high_12h, 1)
    low_prev = np.roll(low_12h, 1)
    close_prev = np.roll(close_12h, 1)
    high_prev[0] = np.nan  # First value has no previous
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    r4 = close_prev + 1.5 * (high_prev - low_prev)
    r3 = close_prev + 1.1 * (high_prev - low_prev)
    s3 = close_prev - 1.1 * (high_prev - low_prev)
    s4 = close_prev - 1.5 * (high_prev - low_prev)
    
    # === 12h Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    
    # === 6h Indicators: ATR (14-period) for stoploss ===
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (6h)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    vol_median_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20)
    # ATR is already on primary timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14)  # Volume median, ATR
    
    # Track position state and entry price for ATR stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(vol_median_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol_median = vol_median_aligned[i]
        r4 = r4_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        atr = atr_14[i]
        
        # Get current 12h volume for volume spike filter
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        current_vol_12h = vol_12h_aligned[i]
        
        # Volume spike filter: current 12h volume > 1.3x median volume
        volume_spike = current_vol_12h > (vol_median * 1.3)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price reaches or falls below R3 (mean reversion)
            # OR ATR stoploss hit (2.5 * ATR below entry)
            if price <= r3 or price <= entry_price - 2.5 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price reaches or rises above S3 (mean reversion)
            # OR ATR stoploss hit (2.5 * ATR above entry)
            if price >= s3 or price >= entry_price + 2.5 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above R4 with volume spike
            if price > r4 and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below S4 with volume spike
            elif price < s4 and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_VolumeSpike1.3x_ATRTrail2.5_v1"
timeframe = "6h"
leverage = 1.0